# DDPG_Fetch
Exploring the performance of Prioritized Experience Replay (PER) with the DDPG+HER scheme on the Fetch Robotics Environemnt

Plots for Mean Success Rates for different Fetch Environments

<p float="middle">
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/plots/all_plots_fr.png?raw=true" width="400" />
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/plots/all_plots.png?raw=true" width="400" /> 
 
</p>

<p float="middle">
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/plots/all_plots_fp.png?raw=true" width="400" />
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/plots/all_plots_fs.png?raw=true" width="400" />
</p>

Performance Plots when varying the alpha parameter on PER
<p float="middle">
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/plots/alpha_plots_fp.png?raw=true" width="400" />
  <img src="https://github.com/sush1996/DDPG_Fetch/blob/master/plots/alpha_plots_fs.png?raw=true" width="400" />
</p>
* Correction: The plot on the right is for FetchSlide but has been mistakenly labelled as FetchPush</br>


Addition of PER along with finetuning the alpha parameter boosts its performance.</br>

The inclusion of the PER algo within the DDPG-HER framework can be done in many ways, it could give greater performance boosts if combined well.
(The integration of PER in this code isn't perfect, just something I tried out over a weekend) </br>

Use the command below to start training. (Avoid using sudo, if you get an "EXPORT LIBRARY.. .bashrc" error) 

```
mpirun -np 19 python3 train.py
```

# compared with baselines~

In https://github.com/openai/baselines/tree/master/baselines/her, 
if we run:

>mpirun -np 19 python -m baselines.run --num_env=2 --alg=her

We will get success rate about 1 at 10 epochs for FetchPush-v1.



# pytorch-gpu vs pytorch-cpu with "mpirun -np 1 python train.py":

pytorch-gpu:
>[2021-02-10 18:05:54.221622] epoch is: 0, eval success rate is: 0.000
[2021-02-10 18:06:33.721338] epoch is: 1, eval success rate is: 0.100
[2021-02-10 18:07:14.687354] epoch is: 2, eval success rate is: 0.100
[2021-02-10 18:07:55.489233] epoch is: 3, eval success rate is: 0.300

pytorch-cpu:
>[2021-02-10 18:08:24.399713] epoch is: 0, eval success rate is: 0.100
[2021-02-10 18:11:02.360476] epoch is: 1, eval success rate is: 0.100
[2021-02-10 18:13:39.896622] epoch is: 2, eval success rate is: 0.000
[2021-02-10 18:16:15.620439] epoch is: 3, eval success rate is: 0.000

Time consumption ratio ≈ 1:5

