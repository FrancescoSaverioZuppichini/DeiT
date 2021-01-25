# DeiT: Data-efficient Image Transformers
**Transformers go brum brum**

Hi guys! Today we are going to implement [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) a new method to perform knowledge distillation on Vision Transformers called DeiT.

You will soon see how elegant and simple this new approach is.


Code is [here](https://github.com/FrancescoSaverioZuppichini/DeiT), an interactive version of this article can be downloaded from [here](https://github.com/FrancescoSaverioZuppichini/DeiT/blob/main/README.ipynb).

DeiT is available on my new computer vision library called [glasses](https://github.com/FrancescoSaverioZuppichini/glasses)

Before starting I **highly** recommend first have a look at [Vision Transformers](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632)

## Introduction

Let's introduce the DeiT models family by having a look at their performance

<img src="https://github.com/FrancescoSaverioZuppichini/DeiT/blob/main/images/DeiTTable.png?raw=true" width="600">

Focus your attention on *ViT-B* and *DeiT-S*. As you can see, their smallest model has + 4-5% and is 100x faster than the bigger *ViT-B*. **How it is possible?**

### Knowledge Distillation

(The paper has a very good summary section about this topic, but I will go fast)

Knowledge Distillation is a training technique to teach a *student* model to match a *teacher* model predictions. This is usually used to, starting from a big model as a *teacher*, produce a new smaller *student* model yielding better performance than training the *student* model from scratch. 

There different types of distillation techniques, in this paper they used what is called hard-label distillation. The idea is to use both the real target $y$ and the target produced by the *teacher* $y_t=\text{argmax}_cZ_t(c)$.

$$
\mathcal{L}_{\text {global }}^{\text {hardDistill }}=\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y\right)+\frac{1}{2} \mathcal{L}_{\mathrm{CE}}\left(\psi\left(Z_{s}\right), y_{\mathrm{t}}\right)
$$

 Where $Z_s$ and $Z_t$ are the logits of the student and teacher model respectively, $\psi$ is the sofmax function.

The loss will penalize the student when it misclassifies real target and the teacher target. This is important because they are not always the same. The teacher could have made some mistake or the picture may have been augmented heavily and thus the target has changed.

Interestingly, the best results were archived when they used a convnet ([regnet](https://arxiv.org/abs/2003.13678)) as a teacher, not a transformer.

<img src="https://github.com/FrancescoSaverioZuppichini/DeiT/blob/main/images/DistillationTableHard.png?raw=true" width="600">

It is called **hard** because the student depends on the hard labels of the teacher. In PyTorch, this can be implemented by


```python
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs: Tensor, outputs : Tensor, labels: Tensor) -> Tensor:
        
        base_loss = self.criterion(outputs, labels)

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs, teacher_labels)
        
        return 0.5 * base_loss + 0.5 * teacher_loss
    
# little test   
loss = HardDistillationLoss(nn.Linear(100, 10))
_ = loss(torch.rand((8, 100)), torch.rand((8, 10)), torch.ones(8).long())
```

### Attention Distillation

<img src="https://github.com/FrancescoSaverioZuppichini/DeiT/blob/main/images/DistillationAttention.png?raw=true" width="600">

ViT employs the **class token** to make its final prediction. Similarly, we can add a **distillation token** that is used to make a second prediction; this second prediction is in the second part of the loss. The authors reported that class and distillation token converges to a very similar vector, as expected because the teacher prediction is similar to the targets, but still not identical.

We can easily modify our loss:


```python
from typing import Union

class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inputs: Tensor, outputs: Union[Tensor, Tensor], labels: Tensor) -> Tensor:
        # outputs contains booth predictions, one with the cls token and one with the dist token
        outputs_cls, outputs_dist = outputs
        base_loss = self.criterion(outputs_cls, labels)

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs_dist, teacher_labels)
        
        return 0.5 * base_loss + 0.5 * teacher_loss
```

Easy!


### Distillation Token

Now we have to add the `dist` token to our model. DeiT is just a normal ViT with this additional token, so I can recycle the code from my [ViT Tutorial](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632). 

This new token, as the class token, is added to the embedded patches.


```python
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        # distillation token
        self.dist_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        dist_tokens = repeat(self.dist_tokens, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
```

### Classification Head

Transformers use the class token to make a prediction, nothing new. In our case, we also use the distillation token to make a second prediction used in the teacher loss.

We also have to change the head to return both predictions at training time. At test time we just average them.


```python
class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):       
        super().__init__()

        self.head = nn.Linear(emb_size, n_classes)
        self.dist_head = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        x_head = self.head(x)
        x_dist_head = self.dist_head(x_dist)
        
        if self.training:
            x = x_head, x_dist_head
        else:
            x = (x_head + x_dist_head) / 2
        return x

```

Then, it follows the same ViT code I used in my [previous article](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632)


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
```

Finally our model looks like:


```python
class DeiT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
```

To train, we can use a bigger model (ViT-Huge, RegNetY-16GF ... ) as *teacher* and a smaller one (ViT-Small/Base) as *student*. The training code looks like this:



```python

ds = ImageDataset('./imagenet/')
dl = DataLoader(ds, ...)

teacher = ViT.vit_large_patch16_224()
student = DeiT.deit_small_patch16_224()

optimizer = Adam(student.parameters())
criterion = HardDistillationLoss(teacher)

for data in dl:
    inputs, labels = data
    outputs = student(inputs)
    
    optimizer.zero_grad()
    
    loss = criterion(inputs, outputs, labels)
    
    loss.backward()
    optimizer.step()
    
```

I am not facebook so I don't have a couple of hundred GPUs lying around so I cannot train these models on ImageNet, but you get the idea! The paper has tons of experiments and I suggest you have a look at it if you are curious.

### Conclusion

In this article, we have seen how to distill knowledge from vision transformer using a new technique.

By the way, I am working on a new computer vision library called [glasses](https://github.com/FrancescoSaverioZuppichini/glasses), check it out if you like

Take care :)

Francesco
