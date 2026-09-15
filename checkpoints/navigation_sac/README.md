# Bundled navigation checkpoint

This directory contains the single published SAC training state:

- `sac_actor.pth` — navigation actor used for evaluation and deployment;
- `sac_critic.pth` and `sac_critic_target.pth` — critics for resumed training;
- `sac_log_alpha.pth` — learned entropy temperature;
- `sac_metadata.json` — training provenance and evaluation conditions.

The checkpoint was trained for 10,000 episodes on the baseline SAC code. Its actor accepts a
56-value observation (47 current features plus three previous 3D commands) and emits
three normalized velocity commands.

See the root README for the reproducible benchmark command and measured result.
