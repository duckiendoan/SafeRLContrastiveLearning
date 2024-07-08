# Safe reinforcement learning with constrastive representation learning
This repository contains the code for a safe reinforcement learning method which uses contrastive learning to detect unsafe states, and then employs a Q-network to interfere with unsafe action execution. The Q-network is trained as per [Karimpanal et al. [2020]](https://arxiv.org/abs/1909.04307). All experiments are conducted on [Minigrid](https://github.com/Farama-Foundation/Minigrid/) environments.
# Codebase structure
The training scripts are based on [CleanRL](https://github.com/vwxyzjn/cleanrl)
- `dqn_minigrid.py`: train a vanilla DQN on Minigrid
- `dqn_minigrid_ae.py`: train a deep auto-encoder (AE) on image observations
- `dqn_minigrid_ae_exp.py`: train a deep AE on image observations with contrastive learning objective
- `dqn_minigrid_safe.py`: train a DQN with a pretrained Q-network to bias against unsafe exploration. This implements Algorithm 2 in [Karimpanal et al. [2020]](https://arxiv.org/abs/1909.04307)
- `dqn_minigrid_safe_exp.py`: train a DQN with a pretrained AE + Q-network to bias against unsafe exploration but maintain exploration (our algorithm)
- `learn_pseudo_Q_minigrid.py`: train Qp function as per [Karimpanal et al. [2020]](https://arxiv.org/abs/1909.04307)
- `visualize_embeddings_ae.py`: visualize the latent representation learned by a pretrained AE
# Experiments
To run experiments, first install dependencies
```shell
pip install -r requirements.txt
```
The experiments are conducted on 3 environments: `MiniGrid-LavaCrossingS9N1-v0, MiniGrid-LavaCrossingS9N2-v0, MiniGrid-LavaCrossingS9N3-v0`. The following example illustrates how to run the algorithm for `MiniGrid-LavaCrossingS9N1-v0`
- Step 1 (Optional): Train optimal Q networks on environments with the same observation space
```shell
for env_id in 'MiniGrid-LavaCrossingS9N1-v0' \
            'MiniGrid-LavaCrossingS9N2-v0' \
            'MiniGrid-SimpleCrossingS9N1-v0' \
            'MiniGrid-SimpleCrossingS9N2-v0'
do
    python dqn_minigrid.py --env_id $env_id \
    --reseed --save-model --total-timesteps 200000 --exploration-fraction 0.9
done
```
Then copy all `.cleanrl_model` files from `runs` folder to a new folder called `qpriors`. This repository already contains these priors, so this step can be ignored.
- Step 2: Train a Q-network to bias against unsafe exploration
```shell
python learn_pseudo_Q_minigrid.py --env_id MiniGrid-LavaCrossingS9N1-v0 \
--reseed --save-model --total-timesteps 250000
```
Then go to the output folder of this run and save the path of the model. A pretrained model can be downloaded [here](https://www.dropbox.com/scl/fi/zljcj7tpxy5wxbkaxyxrx/safeqnetwork.cleanrl_model?rlkey=uvfilstfb6k62ot6fpazrasbs&st=vimktlvp&dl=0).
- Step 3: Train an auto-encoder (AE)
```shell
python dqn_minigrid_ae_exp.py --env_id MiniGrid-LavaCrossingS9N1-v0 \
--no-reseed --save-model --total-timesteps 200000 --exploration-fraction 4.0 \
--no-plot-state-heatmap --ae-batch-size 128 --min-latent-distance 10.0 --latent-dist-coef 0.5
```
- Step 4: Use the pretrained AE and Q-network to bias against unsafe exploration
```shell
python dqn_minigrid_safe_exp.py \
--env_id MiniGrid-LavaCrossingS9N1-v0 --safe-q ./safeqnetwork.cleanrl_model \
--ae-path runs/$RUN_NAME/dqn_minigrid_ae_exp_ae.pt \
--reseed --save-model --total-timesteps 150000 --exploration-fraction 0.9 \
--max-latent-dist 2.5 --safety-batch-size 10 --safety-threshold -0.00 --prior-prob 0.95
```
where `RUN_NAME` is the name of the output folder the step 3.