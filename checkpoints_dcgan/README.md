# Pesos pré-treinados — DCGAN (`aulas_praticas/GAN.ipynb`)

Checkpoints prontos para usar com `LOAD_PRETRAINED = True` no notebook, para quando
o tempo da aula não der para treinar até convergir.

## Arquivos

| Arquivo | O que é |
|---|---|
| `generator_epoch{10,25,40,55}.pt` | `state_dict` do `Generator` em 4 pontos do treino (~50 MB cada) |
| `generator_pretrained.pt` | cópia do checkpoint da época 55, o que a seção de Teste carrega |
| `discriminator_pretrained.pt` | `state_dict` do `Discriminator` correspondente (~11 MB) |
| `samples_epoch*.png` | grade 4x4 gerada em cada época salva |
| `Learning_curves.png` | curvas de perda do pré-treino |

Os checkpoints por época existem para a célula **"Evolução do gerador ao longo do treino"**:
ela carrega todos os `generator_epoch*.pt` do diretório e gera **sempre a partir do mesmo `z`**,
então cada linha da figura é uma época e cada coluna é um `z` fixo — dá para ver o mesmo rosto
saindo de um borrão e ficando nítido.

## Como foram treinados

- **Arquitetura**: exatamente as classes `Generator` / `Discriminator` do notebook,
  sem nenhuma alteração — por isso o `load_state_dict` funciona direto.
- **Latente**: `dim(z) = INPUT_SHAPE = 64`, `Z_train ~ normal` (bate com
  `DISTRIBUTION_train = 'normal'`).
- **Otimização**: igual ao notebook — Adam `lr=1e-4`, `betas=(0.5, 0.999)`, BCE,
  label smoothing, batch 32.
- **Dados**: 30.000 rostos do CelebA (`nielsr/CelebA-faces` no HuggingFace),
  center-crop quadrado e resize para 64x64, normalizados em [-1, 1].
  O dataset do Kaggle `ai-face-dataset-3000-images` exige credenciais, então o
  pré-treino usou esse substituto público — mesmo domínio (rostos 64x64),
  10x mais imagens, o que ajuda a convergir.
- **Épocas**: 55 (ver `training_summary.txt`).

Script de treino: `train_gan.py` (neste mesmo diretório de trabalho).

## Como usar no Kaggle

1. Crie um Kaggle Dataset (ex.: `dcgan-faces-pretrained`) e suba os dois `.pt`.
2. No notebook: **Add Input → Datasets →** selecione esse dataset.
3. Na célula de hiperparâmetros:
   ```python
   LOAD_PRETRAINED = True
   CHECKPOINT_DIR  = '/kaggle/input/dcgan-faces-pretrained'  # confira o path real em /kaggle/input/
   ```
4. Rode tudo. A célula de treino é pulada e a seção de **Teste** carrega os pesos
   e gera as imagens para `Z_train` vs `Z_test`.

## Uso local

```python
generator = Generator(64, 0.2)
generator.load_state_dict(torch.load('generator_pretrained.pt', map_location='cpu'))
generator.eval()
```
