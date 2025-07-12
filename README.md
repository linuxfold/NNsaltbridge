# NNsaltbridge
Simple Feed Forward Neural Net to Predict Artificial Salt Bridges

Given a PDB or CIF structure, predict which 2 residues could be mutated to create an artificial salt bridge.

You can install and predict directly (skipping build/train) by supplying the included sb_model.pt, which was trained on the same mmcif files that are used as part of the AF3 database (but includes newer files up to 5/24/25) and the compressed AFDB. 


**1 | What the script does**

**build**	

Parse every .mmCIF in a directory, pull out all salt bridges (inter- & intra-chain) and generate matching negative examples.	Uses gemmi for blazing-fast CIF access and multiprocessing to scale across cores.

**train**	

Learn to score salt bridge compatibility.	A lightweight PyTorch feed-forward network (3 numeric features → 16→8→1). Default training loop + early model checkpointing.

**predict**	

Scan a new structure, enumerate residue pairs that could form bridges, and rank them by probability they could form a salt bridge once mutated to Arg/Lys/Glu/Asp.	CA-distance cutoff defaults to 8 Å; adjust with --cutoff. Results land in predictions.csv (chain, residue numbers, prob).

All three stages are wrapped as CLI sub-commands, so one file drives the whole pipeline.


**2 | Installation:**

    git clone https://github.com/linuxfold/NNsaltbridge
    cd NNsaltbridge
    
    conda create -n sbridge python=3.11 gemmi pytorch pandas numpy tqdm
    conda activate sbridge

**3 | Build & Extract Data**

      python NNsaltbridge.py build \
      --data_dir /data/pdb-mmCIF \
      --out_csv saltbridges.csv \
      --nproc 32        # adapt to your CPU budget

Streams rows as it parses → you can Ctrl-C anytime and keep progress.

Input may be mixed .cif, .pdb, or .gz.      

**4 | Train**

      python NNsaltbridge.py train \
      --dataset   saltbridges.csv \
      --model     sb_model.pt     \
      --epochs    1000

**5 | Predict**

      python saltbridge_predictor.py predict \
      --model      sb_model.pt \
      --structure  6sc2.cif    \
      --cutoff     8           # Å, charged-atom filter
      --top_k      50          # 0 = no limit
      
Outputs sb_preds.csv with:

chain1 res1 chain2 res2 prob
A      102  B      44   0.94

**6 · Design new bridges (mutation suggestions)**

Intra + inter-chain

      python saltbridge_predictor.py design \
      --model sb_model.pt \
      --structure 6sc2.cif \
      --top_k 100
      
Inter-chain only

      python saltbridge_predictor.py design_inter \
      --model sb_model.pt \
      --structure 6sc2.cif \
      --top_k 100
      
Outputs
chain1 res1 mut1 chain2 res2 mut2 prob, e.g.
A 67 ASP   B 102 LYS 0.92
