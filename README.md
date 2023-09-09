![Logo](https://cdn-images-1.medium.com/v2/resize:fill:36:36/1*GWoejvreNB-w4joRbxLtog.png)

# Tree Defects Diagnosis with AI
*Project of Forestree, Remote Sensing and Forestry*

Implementing an AI-based tree defect diagnosis tool can help enhance tree management services for the public. After storms, trees often become unstable due to structural issues like decay and pests/diseases. However, people may not report defects to tree crews due to lack of expertise in professional terminology. While global AI models exist, but they do not align with standards set by the local Tree Management Office (TMO).

Therefore, Forestree is developing a predictive system to accurately scan trees, identify defects based on TMO standards, and report issues. This user-friendly AI tool will empower the public to effectively monitor tree conditions. It aims to reduce response times for tree crews by facilitating earlier reporting. Additionally, involving citizens through AI will lighten the workload for government departments and improve overall management efficiency and effectiveness by catching issues sooner.

The deep learning process make use of the Swin Transformer architecture. It is designed to classify images into one of 30 classes according to the Tree Management office standard.
## Demo

![Logo](https://lh4.googleusercontent.com/CFcHwFvFrHvBdAG3x3AyCuMQFXR8AkQYiQeMlUxvot2a0gNPo38KfbBFsDIzLqkoM-XqRce0bKxkdNzE28Iz_H_XIDDm0AyPOAsmmPUK1-tcM34kJ6vzpEUIHGzw6UvKMlMiXALgEP-GRDWhWSTgPamlwidd2A)

![Logo](https://lh4.googleusercontent.com/RccW0q79coexQd197PvamludHe1lAcNptEqlPiHUxhKZCrrmPbD13cRupl8G2jAlaajciEFw-oH8dYi7Uu_5WLqn8jBOHJjrUMOql1HT9T1zHHRPTvhRnte82AsYQabIVrsgN8avt6__bYbbGRhnWBpXGghJcw)


## Installation

Clone this Tree vision package 

```
!git clone https://github.com/ottoykh/Tree-vision.git
```
This Project directory: 
```
Tree-vision/
   TreeVision/
      TreeAI/
        __init__.py
        api.py
    __init__.py
```
Direct to the pre-write code file with: 
```
cd /content/Tree-vision/TreeVision/TreeAI
```
## Usage/Examples

```python
from Diagnosis import TreeAI, TreeAI_Batch

TreeAI("Your Target Image")
TreeAI_Batch("Your Target Folder","Result Output.csv")

```


## Authors

- [Yu Kai Him Otto](https://www.github.com/ottoykh)

