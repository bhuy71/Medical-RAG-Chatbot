# End-to-end-Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: git clone repo_link
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Download the quantize model from the link provided in model folder & keep the model in the model directory:
# You have to download the large language model named: llama-2-7b-chat.Q4_0.gguf from this link:https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main and put it in folder model.


```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone


