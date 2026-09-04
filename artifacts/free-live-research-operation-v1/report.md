# free-live-research-operation-v1

ARTIFACT_SHA=f2060dde33c41a6cf6f936d15224a4c7d23e452785d370314077e3e30ee68cf2

## Independent Answers

OPERATION=YES
ML_DATASET=NO

## Metrics

1. BASE_MAIN_SHA="1e785ee82a20fdb893f213dee190238dc1558e44"
2. HEAD_SHA="1e785ee82a20fdb893f213dee190238dc1558e44"
3. artifact SHA="f2060dde33c41a6cf6f936d15224a4c7d23e452785d370314077e3e30ee68cf2"
4. operational sources=["ROSN_ROSNEFT_PRESS_RELEASES_RSS_EXACT_LIVE_V1", "YDEX_YANDEX_IR_PRESS_RELEASES_RSS_EXACT_LIVE_V1"]
5. healthy sources=["ROSN_ROSNEFT_PRESS_RELEASES_RSS_EXACT_LIVE_V1", "YDEX_YANDEX_IR_PRESS_RELEASES_RSS_EXACT_LIVE_V1"]
6. degraded sources=[]
7. real bounded polls=4
8. HTTP requests=4
9. newly discovered publications=6
10. duplicates=6
11. revisions=0
12. total shadow events=6
13. semantic-ready=6
14. UNKNOWN count/rate={"count": 3, "rate": "0.500000"}
15. feature attempts=6
16. feature-ready=0
17. feature blocked=6
18. retryable feature blockers=0
19. permanent feature blockers=6
20. collector restart test="covered by state/idempotency tests"
21. idempotency test="covered by duplicate replay tests"
22. sealed verify="run `python -m apps.cli.live_issuer_verify_seal`"
23. live outcomes read=0
24. targets computed=0
25. post-event reads=0
26. model trained=false
27. broker mutations=0
28. LIVE_RESEARCH_OPERATION_STATUS="READY"
29. ML_V2_DATASET_STATUS="BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY"
30. exact next action="Leave free official live research collector running on ROSN/YDEX; keep ML v2 blocked until issuer diversity improves."
