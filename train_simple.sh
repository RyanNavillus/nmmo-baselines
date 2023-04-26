python -O -m main \
--model.arch=simple \
--env.num_teams=8 \
--env.team_size=1 \
--rollout.num_envs=1 \
--rollout.num_buffers=1  \
--rollout.num_steps=128 \
--wandb.entity=daveey \
--wandb.project=nmmo \
--checkpoint_dir=/fsx/home-daveey/checkpoints/simple.1 \
--resume_from=latest \
--train.num_steps=100000000
