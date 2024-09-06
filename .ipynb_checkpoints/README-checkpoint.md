# aiMusing
Musing over AI

- *large_language_hashmap.ipynb* -> Trying to understand if LLM approximates a high dimensional hash map lookup
- *is_llm_thinking_in_english.ipynb* -> Checking the tokens of the hidden layers of LLM for Tamil to Hindi language translation. The model seems to first translate the contents to english and then to target language as seen by the token predictions of the hidden layers.
- *whats_llm_predicting_internally.ipynb* -> Continuation of the is_llm_thinking_in_english.ipynb to check the top-k predicted tokens by hidden layers
- *is_maths_also_language.ipynb* -> Checking if translation has an effect on the reasoning abilities of the model. Trying out simple arithmetic in Devnagari numbers and checking what the internal state depicts.
- *journey_of_information.ipnb* -> Try to understand what is even information, how does language capture it and how good are embeddings at capturing it, are there any issues?


### Notes
- The code here takes any inspiration from [Do Llamas Work in English? On the Latent Language of Multilingual Transformers
](https://arxiv.org/abs/2402.10588) [(Code)](https://github.com/epfl-dlab/llm-latent-language) and [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) [(Code)](https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg?usp=sharing#scrollTo=i1vLDXdxzrE5)
- Code can also be ran on M2 Macbook pro with MPS
- Hindi to Tamil translation data was created manually
- Weights for the llama-2 were downloaded from: https://llama.meta.com/llama-downloads/
- Used the following mechanism to convert the weights to a Huggingface compatible format
```
git clone git@github.com:huggingface/transformers.git
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ~/src/open/llama --output_dir ~/src/open/llm-latent-language --model_size 7B  --llama_version 2
```
- Setup code to run this on remote GPU
```
ssh-keygen -t rsa -b 4096 -C <github_email>
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/id_rsa
cat /root/.ssh/id_rsa.pub
git clone git@github.com:meta-llama/llama.git
git clone git@github.com:huggingface/transformers.git
cp /home/root/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py /home/root
rm -rf /home/root/transformers
# ^ To save space on GPU
git clone git@github.com:epfl-dlab/llm-latent-language.git
mkdir /home/root/llm-latent-language/Llama-2-7B-hf
cd llama
./download.sh
mv /home/root/llama/llama-2-7b/* /home/root/llama
python /home/root/convert_llama_weights_to_hf.py --input_dir /home/root/llama --output_dir /home/root/llm-latent-language/Llama-2-7B-hf --model_size 7B  --llama_version 2 && rm -rf /home/root/llama
pip install -U bitsandbytes 
```
