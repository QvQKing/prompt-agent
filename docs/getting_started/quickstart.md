### Quick Start: Try Default Search Tool on HotpotQA
#### 1. Install `FlagEmbedding` and `faiss`
```bash
pip3 install FlagEmbedding
pip3 install faiss-cpu
```

#### 2. Download and preprocess HotpotQA dataset
```bash
# Create data directory
mkdir -p data/hotpotqa

# Run the preprocessing script
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa
```

This script will:
- Download the HotpotQA dataset directly from the source
- Process the data into the format required by Agent-R1
- Save the processed data as train.parquet and validation.parquet in the specified directory

#### 3. Build hotpotqa search index
```bash
# Download the corpus file (gzipped)
mkdir -p data/corpus/hotpotqa
wget https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz

# Extract the gzipped file
gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl

# Process the corpus and build the search index
cd scripts/hotpotqa_search
python process_hotpotqa.py
cd ../../
```

This script will:
- Load the corpus data
- Generate embeddings using the BAAI/bge-large-en-v1.5 model
- Build a FAISS index for efficient similarity search
- Save the embeddings and index files in the data/corpus/hotpotqa directory

#### 4. Run PPO/REINFORCE++/GRPO training with Qwen2.5-1.5B-Instruct
```bash
# Run the PPO training script
cp examples/trainer/run_ppo_hotpotqa.sh ./
bash run_ppo_hotpotqa.sh
# Run the REINFORCE++ training script
cp examples/trainer/run_rpp_hotpotqa.sh ./
bash run_rpp_hotpotqa.sh
# Run the GRPO training script
cp examples/trainer/run_grpo_hotpotqa.sh ./
bash run_grpo_hotpotqa.sh
```

### 5. Results on HotpotQA

#### PPO

![ppo](../../image/ppo.jpg)

#### REINFORCE++

![rpp](../../image/rpp.jpg)

#### GRPO

![grpo](../../image/grpo.jpg)

We can see that the model (Qwen2.5-1.5B-Instruct) effectively learns to think and then invoke the tool in multiple rounds when faced with challenging multi-hop questions, ultimately achieving improved the EM results. The effectiveness of different reinforcement learning algorithms varies, but the general trend is the same.

Notably, our experiments reveal a striking correlation: EM scores, number of tool calls (turns), and final response length all display consistent trends across training. This demonstrates a novel dimension of scaling laws—one that relates to the frequency of agent-environment interactions. As the agent learns to interact more effectively with its environment through multiple tool calls, performance improves proportionally, suggesting that the ability to engage in multiple rounds of environment interaction may be as crucial to agent performance as traditional scaling factors.