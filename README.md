# Learning, Intelligence + Signal Processing (LISP) Lab - Dartmouth

<img width="1288" alt="Screenshot 2024-05-25 at 2 21 56 AM" src="https://github.com/aniketxdey/lisp/assets/168318141/aafccad9-ce0b-473e-bbb3-c2f2e04f44cf">

Research conducted under advisory of Professor Peter Chin at LISP Machine Learning Lab. 

## Codebase Guide:
1. `asr-model-attack` - An Attack to Reveal Voice Samples in Distributed Speech Recognition Models
    - Algorithms & training available in `/src/notebooks`
2. `mamba-ESI` - Mamba-ESI: Introducing Embedding Search to Selective SSMs
    - Architecture & code available in `/mamba_ssm/mixer_seq_simple.py`

## An Attack to Reveal Voice Samples in Distributed Speech Recognition Models
#### Peter Chin (Advisor), Minh Bui (PI), Paul Cherian, Mary Wood, Aniket Dey

### Abstract
Automatic Speech Recognition (ASR) systems are a class of technologies that have been deployed in various speech-to-text applications, enabling seamless communication between humans and computers. Modern ASR systems employ a multitude of deep learning frameworks to analyze audio input data and identify speech patterns. These frameworks are commonly trained in Federated Learning (FL) environments, which use a system of devices to collectively train a central model while keeping individual data private by only transmitting gradients. In this paper, we present a method to “leak” this private training data used to train ASR systems by employing Gradients Matching for the CTC-Loss Function. We find this method to perform with high accuracy in multiple experiment with both simple recognition models as well as DeepSpeech2, a leading deep learning framework for ASR.

### 2024 Wetterhahn Science Symposum Poster:
![Wetterhahn Science Symposium (1)](https://github.com/aniketxdey/lisp/assets/168318141/f18d3f1e-ea69-4671-8697-3675f63843e1)
[Wetterhahn Science Symposium (1).pdf](https://github.com/aniketxdey/lisp/files/15441960/Wetterhahn.Science.Symposium.1.pdf)

## Mamba-ESI: Introducing Embedding Search to Selective SSMs
#### Peter Chin (Advisor), Yu-Wing Tai (Advisor), Abhinav Reddy, Aniket Dey

### Abstract: 
Sequence modeling has made significant strides with the advent of powerful architectures like Transformers. However, efficiently handling long input sequences, such as entire textbooks, remains a challenge. Existing approaches often struggle to capture and utilize relevant information from distant parts of the sequence, leading to sub-optimal performance on tasks like question answering. In this paper, we propose Mamba-ESI, an extension to the Mamba sequence model that introduces an embedding search and injection mechanism to selectively amplify relevant information from the input sequence. By computing embedding for the question and input tokens, Mamba-ESI performs a similarity search to identify the most relevant tokens and injects their corresponding hidden states into the current hidden state of the model, effectively performing kernel trick on the fly. This approach enables Mamba-ESI to dynamically attend to pertinent information without the need to store and process the entire memory tensor. We conduct experiments on a range of long-sequence tasks, including textbook question answering, and demonstrate that Mamba-ESI achieves competitive performance while maintaining significant memory and computational efficiency compared to state-of-the-art methods. Furthermore, we analyze the impact of various design choices, such as the embedding function and injection mechanism, on the model's performance. Our results highlight the effectiveness of embedding search and injection in enabling efficient and accurate sequence modeling for long input sequences, paving the way for more scalable and versatile sequence models.

### Mamba ESI Benchmark Performance:
<img width="691" alt="Screenshot 2024-11-07 at 5 34 48 PM" src="https://github.com/user-attachments/assets/ee3c52d9-6c0d-4fa2-9c22-7b77aaafad74">

### Project README.md
(https://github.com/aniketxdey/lisp/tree/main/mamba-ESI)

### Project Presentation to LISP Lab Postdocs:

[mambaESI - aniket dey & abhinav reddy.pdf](https://github.com/user-attachments/files/17670232/mambaESI.-.aniket.dey.abhinav.reddy.pdf)






