# Relatório de Pesquisa: Artigos e Códigos para o Pipeline de Vetorização

> **Data da Pesquisa:** 02/09/2026  
> **Objetivo:** Identificar os melhores artigos da literatura do ArXiv, códigos abertos e técnicas de ponta para aprimorar o pipeline de vetorização de imagem e vídeo localizado em [`engine/`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine).  
> **Relatório HTML Interativo Completo:** [agentArchive/latest_report.html](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/agentArchive/latest_report.html)  
> **Gráfico de Tendências ArXiv:** [agentArchive/trend_chart.png](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/agentArchive/trend_chart.png)

---

## 1. Contexto do Pipeline Atual (`engine/`)

O repositório atual possui um mecanismo modular de vetorização de imagens e animação de vídeo em SVG:
* [`engine/vectorizer.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/vectorizer.py): Integração nativa com o **VTracer** para ajuste de curvas Bézier (*Bézier curve fitting*), pré-processamento híbrido e consolidação de cor.
* [`engine/preprocessor.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/preprocessor.py): Filtragem bilateral e Mean-Shift para estilo Cel-Shading/flat.
* [`engine/palette.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/palette.py): Extração de paleta mestra de vídeo e quantização perceptual em espaço de cor CIELAB.
* [`engine/flipbook.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/flipbook.py): Geração de SVG animado (*Flipbook*) com controle de FPS, classes CSS e cels de animação.
* [`engine/svg_parser.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/svg_parser.py): Parser de caminhos `<path>`, cálculo de `bounding boxes`, detecção de fundo e arredondamento de coordenadas.

---

## 2. Configuração do ArXiv-Agent (`agentArchive/`)

### 🛡️ Alerta de Segurança Resolvido
* O link inicial (`https://github.com/azizkode/ArXiv-Agent`) continha instaladores zip binários maliciosos (`Ar-Agent-Xiv-2.1.zip`).
* O agente foi configurado utilizando a base de código oficial e limpa de [**`banjiuyufen/ArXiv-Agent`**](https://github.com/banjiuyufen/ArXiv-Agent).

### 🛠️ Melhorias Técnicas Adicionadas ao Agente
1. **Fallback de Relatório Local:** Geração automática de `latest_report.html` e `trend_chart.png` mesmo sem credenciais SMTP no `.env`.
2. **Scoring Heurístico Vetorial:** Algoritmo de ranqueamento (0 a 10) baseado em proximidade semântica com o pipeline (curvas Bézier, R2V, SVG, DiffVG, animação, consistência temporal).
3. **Ordenação por Relevância:** Suporte a `arxiv.SortCriterion.Relevance` para buscar artigos sem restrição artificial de data recente.
4. **Deep Scan de LaTeX:** Inspeção automática de fontes compactadas do ArXiv para detecção de conferências (CVPR, ICCV, IEEE, AAAI, ICML, etc.) e extração de links do GitHub.

---

## 3. Top Artigos, Repositórios e Códigos Identificados

### A. Animação de Vídeo e Consistência Temporal de Traços

#### 📄 [Vector Prism: Animating Vector Graphics by Stratifying Semantic Structure](https://arxiv.org/abs/2512.14336)
* **Autores:** Jooyeol Yun, Jaegul Choo (CVPR 2025/2026)
* **ArXiv ID:** `2512.14336v1` | [PDF](https://arxiv.org/pdf/2512.14336v1)
* **Score do Agente:** `9/10`
* **Problema Resolvido:** Em pipelines de vetorização de vídeo como o `flipbook.py`, conversores automáticos fragmentam elementos visuais em dezenas de caminhos (`<path>`) individuais sem hierarquia. Entre um frame e outro, essa ordem varia, causando tremulação (*flickering* visual).
* **Solução:** O *Vector Prism* propõe a **estratificação semântica**: agrupa automaticamente caminhos conectados espacialmente e cromaticamente em grupos lógicos (`<g id="layer_head">`, `<g id="layer_body">`). Isso permite estabilizar a transição entre cels e interpolar frames com consistência de movimento.

---

### B. Raster-to-Vector (R2V) e Poda de Traços Bézier

#### 📄 [MARVEL: Raster Manga Vectorization via Primitive-wise Deep Reinforcement Learning](https://arxiv.org/abs/2110.04830)
* **Autores:** Hao Su, Jianwei Niu, et al. (IEEE)
* **ArXiv ID:** `2110.04830v2` | [PDF](https://arxiv.org/pdf/2110.04830v2)
* **Repositório GitHub:** ⭐ [SwordHolderSH/Mang2Vec](https://github.com/SwordHolderSH/Mang2Vec)
* **Score do Agente:** `10/10`
* **Problema Resolvido:** Arquivos SVG gerados por rasterizadores frequentemente possuem excesso de nós Bézier, curvas colineares redundantes e sobreposição de traços pretos/contornos, inflando o tamanho do SVG.
* **Solução:** O método decompõe a imagem em primitivas de traço e aplica um **mecanismo de poda (*pruning mechanism*)** que elimina curvas duplicadas e simplifica nós intermediários mantendo fidelidade de curvatura máxima com peso mínimo de arquivo.

#### 📄 [Im2Vec: Synthesizing Vector Graphics without Vector Supervision](https://arxiv.org/abs/2102.02798)
* **Autores:** Pradyumna Reddy, Michael Gharbi, Michal Lukac, Niloy J. Mitra (CVPR)
* **ArXiv ID:** `2102.02798v3` | [PDF](https://arxiv.org/pdf/2102.02798v3)
* **Projeto:** [geometry.cs.ucl.ac.uk/projects/2021/Im2Vec](http://geometry.cs.ucl.ac.uk/projects/2021/Im2Vec/)
* **Score do Agente:** `10/10`
* **Solução:** Modelo de R2V que gera curvas Bézier fechadas e preenchimento de cor usando um renderizador raster diferenciável (conceito similar ao DiffVG), ajustando topologia sem supervisão vetorial explícita.

---

### C. Geração Estruturada e Rápida de SVG Colorido

#### 📄 [SVGBuilder: Component-Based Colored SVG Generation](https://arxiv.org/abs/2412.10488)
* **Autores:** Zehao Chen, Rong Pan (AAAI 2025)
* **ArXiv ID:** `2412.10488v4` | [PDF](https://arxiv.org/pdf/2412.10488v4)
* **Repositório Base:** ⭐ [alexandre01/deepsvg](https://github.com/alexandre01/deepsvg) (1.1k+ stars)
* **Score do Agente:** `9/10`
* **Solução:** Gera SVGs coloridos em camadas de forma baseada em componentes (*component-based*), sendo até **604 vezes mais rápido** que abordagens baseadas em otimização iterativa. Demonstra a importância de manter dados de cor e hierarquia de camadas desacoplados.

#### 📄 [Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation](https://arxiv.org/abs/2505.24499)
* **Autores:** Ximing Xing, Ziteng Xue, et al. (CVPR 2025)
* **ArXiv ID:** `2505.24499v2` | [PDF](https://arxiv.org/pdf/2505.24499v2)
* **Score do Agente:** `10/10`
* **Solução:** Introduz o paradigma *Drawing-with-Thought* (DwT) e funções de recompensa híbridas para garantir a validade estrutural de caminhos fechados e estética visual coerente.

---

### D. Processamento e Reconhecimento Direto em SVG (Sem Rasterizar)

#### 📄 [Recognizing Vector Graphics without Rasterization (YOLaT)](https://arxiv.org/abs/2111.03281)
* **Autores:** Xinyang Jiang, Lu Liu, et al. (Microsoft Research)
* **ArXiv ID:** `2111.03281v3` | [PDF](https://arxiv.org/pdf/2111.03281v3)
* **Repositório GitHub:** ⭐ [microsoft/YOLaT-VectorGraphicsRecognition](https://github.com/microsoft/YOLaT-VectorGraphicsRecognition)
* **Score do Agente:** `7/10`
* **Solução:** O YOLaT constrói multigrafos para modelar a informação espacial e estrutural contida diretamente no XML do SVG, permitindo detectar e classificar objetos vetoriais com alta eficiência sem precisar renderizar em pixels.

---

## 4. Plano de Ação e Melhorias para Implementar no Pipeline

### 🚀 Melhoria 1: Poda e Simplificação de Nós Bézier em `svg_parser.py`
* **Inspiração:** Artigo *MARVEL (Mang2Vec)*.
* **Aplicação:** No [`engine/svg_parser.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/svg_parser.py), criar uma função pós-processadora que:
  - Analisa sequências de comandos `C` (curvas cúbicas) e `L` (linhas).
  - Caso três nós de controle consecutivos sejam aproximadamente colineares dentro de uma tolerância angular ($\epsilon < 2^\circ$), funde os segmentos em uma única curva Bézier cúbica equivalente.
  - **Impacto estimado:** Redução de 20% a 35% no tamanho do arquivo SVG gerado pelo VTracer, com curvas mais suaves para animações no navegador.

### 🚀 Melhoria 2: Agrupamento Semântico de Camadas em `flipbook.py`
* **Inspiração:** Artigo *Vector Prism*.
* **Aplicação:** No [`engine/flipbook.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/flipbook.py):
  - Em vez de despejar dezenas de caminhos soltos no `<g id="cel-1">`, usar a proximidade das `bounding boxes` (já calculadas em `compute_path_bbox`) e a proximidade de cor (distância Lab em `palette.py`) para empacotar caminhos relacionados em `<g class="cluster-...">`.
  - **Impacto estimado:** Facilita edição vetorial em ferramentas como Illustrator/Figma e viabiliza interpolação suave de opacidade entre cels.

### 🚀 Melhoria 3: Estabilização de Paleta Temporal para Vídeo
* **Inspiração:** Artigos de restauração de vídeo (*VRT*) e *Im2Vec*.
* **Aplicação:** No [`engine/palette.py`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/engine/palette.py):
  - Atualmente a paleta mestra é calculada por amostragem de frames.
  - Adicionar um histograma de persistência temporal: cores que aparecem em menos de 10% dos frames e possuem distância Lab $< 12$ de uma cor frequente são automaticamente unificadas, eliminando "cores fantasmas" em animações rápidas.

---

## 5. Como Executar Novas Buscas no ArXiv-Agent

Para refazer buscas ou buscar novos temas a qualquer momento:
```powershell
cd c:\Users\izael\Projetos\Python_and_Backend\svg\agentArchive
python code/main.py
```
* As consultas podem ser editadas diretamente no arquivo [`agentArchive/.env`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/agentArchive/.env) no campo `ARXIV_QUERY`.
* O perfil de pesquisa pode ser ajustado em [`agentArchive/user_profile.json`](file:///c:/Users/izael/Projetos/Python_and_Backend/svg/agentArchive/user_profile.json).
