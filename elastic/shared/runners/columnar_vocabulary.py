# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import hashlib
import json
import os

SYSLOG = "logs-system.syslog-*"
REDIS_SLOWLOG = "logs-redis.slowlog-*"
PAGE_SIZE = 2000
TOP_N = 20

_chosen_mac = None


async def _resolve_mac(es):
    """A host.mac term of median frequency among the terms of system.syslog. The terms aggregation orders by document count
    descending and key ascending, so equal data gives the same term in every run. Resolved once and cached."""
    global _chosen_mac
    if _chosen_mac is None:
        response = await es.perform_request(
            method="POST",
            path=f"/{SYSLOG}/_search",
            params={"request_cache": "false"},
            body={"size": 0, "aggs": {"macs": {"terms": {"field": "host.mac", "size": 10000, "order": [{"_count": "desc"}, {"_key": "asc"}]}}}},
        )
        agg = response["aggregations"]["macs"]
        if agg.get("sum_other_doc_count", 0) != 0:
            raise AssertionError("host.mac has more than 10000 terms, the term choice would be incomplete")
        _chosen_mac = agg["buckets"][len(agg["buckets"]) // 2]["key"]
    return _chosen_mac


async def term_count(es, params):
    """Q2: documents of system.syslog that hold the chosen host.mac term, as a search with the request cache off."""
    mac = await _resolve_mac(es)
    response = await es.perform_request(
        method="POST",
        path=f"/{SYSLOG}/_search",
        params={"request_cache": "false"},
        body={"size": 0, "track_total_hits": True, "query": {"term": {"host.mac": mac}}},
    )
    return {"weight": 1, "unit": "ops", "hits": response["hits"]["total"]["value"]}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


async def _composite(es, index, fields):
    sources = [{field: {"terms": {"field": field, "missing_bucket": True}}} for field in fields]
    after = None
    buckets = []
    pages = 0
    while True:
        spec = {"sources": sources, "size": PAGE_SIZE}
        if after is not None:
            spec["after"] = after
        response = await es.perform_request(
            method="POST", path=f"/{index}/_search", params={"request_cache": "false"}, body={"size": 0, "aggs": {"g": {"composite": spec}}}
        )
        agg = response["aggregations"]["g"]
        pages += 1
        buckets.extend((tuple(b["key"][f] for f in fields), b["doc_count"]) for b in agg["buckets"])
        after = agg.get("after_key")
        if after is None or not agg["buckets"]:
            return sorted(buckets, key=lambda b: (-b[1], tuple("" if k is None else str(k) for k in b[0]))), pages


def _tie_aware(full, timed_rows):
    """Compares a timed top N with the complete ranking: groups above the cutoff count must match exactly, groups at the cutoff count
    may be any members of that tie."""
    if len(full) < TOP_N:
        return {"checked": False, "reason": "fewer groups than the limit"}
    cutoff = full[TOP_N - 1][1]
    above = {tuple(b[0]): b[1] for b in full if b[1] > cutoff}
    tie = {tuple(b[0]) for b in full if b[1] == cutoff}
    timed = {tuple(r[:-1]): r[-1] for r in timed_rows}
    return {
        "checked": True,
        "cutoff_count": cutoff,
        "tie_size": len(tie),
        "above_cutoff_equal": {k: v for k, v in timed.items() if v > cutoff} == above,
        "tie_members_valid": all(k in tie for k, v in timed.items() if v == cutoff),
    }


async def validate(es, params):
    """Retrieves every group of Q1, Q3 and Q4 by paging composite aggregations until the server returns no after_key, counts the
    documents of Q2 and writes the result, with the tie-aware check of each timed top 20, to a JSON file on the load driver.
    The file is the evidence of identical results across arms. It is not a timed task."""
    name = params.get("label", "unnamed")
    out_dir = os.path.expanduser(params.get("output-dir", "~/columnar-vocabulary"))
    os.makedirs(out_dir, exist_ok=True)
    mac = await _resolve_mac(es)
    total = (await es.perform_request(method="POST", path="/logs-*/_count"))["count"]
    syslog = (await es.perform_request(method="POST", path=f"/{SYSLOG}/_count"))["count"]
    matching = (await es.perform_request(method="POST", path=f"/{SYSLOG}/_count", body={"query": {"term": {"host.mac": mac}}}))["count"]
    result = {"name": name, "mac": mac, "docs_logs": total, "docs_syslog": syslog, "Q2": {"matching_docs": matching, "selectivity": matching / syslog}}
    timed_queries = {
        "Q1": ("logs-*", ["agent.version", "agent.type", "agent.hostname"], params["q1-query"]),
        "Q3": (SYSLOG, ["host.mac"], f"FROM {SYSLOG} | STATS c=count(*) BY host.mac | SORT c DESC | LIMIT {TOP_N}"),
        "Q4": (REDIS_SLOWLOG, ["redis.slowlog.args"], f"FROM {REDIS_SLOWLOG} | STATS c=count(*) BY redis.slowlog.args | SORT c DESC | LIMIT {TOP_N}"),
    }
    for query_id, (index, fields, esql) in timed_queries.items():
        groups, pages = await _composite(es, index, fields)
        timed = await es.perform_request(method="POST", path="/_query", params={"format": "json"}, body={"query": esql})
        result[query_id] = {
            "groups": len(groups),
            "sum_doc_count": sum(g[1] for g in groups),
            "pages": pages,
            "hash": _digest(groups),
            "tie_check_against_timed_query": _tie_aware(groups, timed.get("values", [])),
        }
    result["Q1"]["complete"] = result["Q1"]["sum_doc_count"] == total
    with open(os.path.join(out_dir, f"validation-{name}.json"), "w") as out:
        json.dump(result, out, indent=2)
    return {"weight": 1, "unit": "ops"}
