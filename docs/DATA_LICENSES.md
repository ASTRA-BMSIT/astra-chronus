# Dataset Licenses

One entry per dataset. **No dataset enters training until its licence entry
exists and is verified** (README section 6). An unverified licence is
recorded as "not yet verified" - never assumed, never copied from a paper's
citation without checking the source.

How to verify: open the dataset's official page or download agreement, read
the licence terms, and paste the licence name plus a link to the source
document. If the terms are unclear or absent, write "unclear - do not use"
and open an issue; that answer is itself a contribution.

| Dataset | Source | Licence | Verified | Link to terms | Notes |
|---|---|---|---|---|---|
| Anti-UAV | public download | not yet verified | – | – | Train and study |
| DUT Anti-UAV | public download | not yet verified | – | – | Train and study |
| Drone-vs-Bird | public download | not yet verified | – | – | Train and study |
| MAV-VID | public download | not yet verified | – | – | Train and study |
| Club footage (own) | recorded by the club | own footage | – | – | Fine-tune and test; log GPS and IMU next to video as ground truth |

Also check before committing to a model: **Ultralytics YOLO weights are
AGPL-licensed** - that matters if we ever publish or productize
(README section 3.1).

Row format: `Dataset | Source | Licence name | date verified | link to terms
| notes`. Verified means someone read the terms on that date and linked
them; a row is written in the same PR that starts using the dataset.
