# GSL4Rec: Session-based Recommendations with Collective Graph Structure Learning and Next Interaction Prediction

This is the code in [GSL4Rec: Session-based Recommendations with Collective Graph Structure Learning and Next Interaction Prediction](<https://dl.acm.org/doi/10.1145/3485447.3512085>) which has been accepted by WWW 2022.

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
```

## Data Process

Prepare the experiment data for Training:

```setup
python data_process.py
```

## Training

To train the model(s) in the paper:

```setup
python main.py
```
> Output: the file "output.tar"

## Citation

If you use the code, please cite following paper,

```latex
@inproceedings{DBLP:conf/www/WeiBBW22,
  author    = {Chunyu Wei and
               Bing Bai and
               Kun Bai and
               Fei Wang},
  editor    = {Fr{\'{e}}d{\'{e}}rique Laforest and
               Rapha{\"{e}}l Troncy and
               Elena Simperl and
               Deepak Agarwal and
               Aristides Gionis and
               Ivan Herman and
               Lionel M{\'{e}}dini},
  title     = {GSL4Rec: Session-based Recommendations with Collective Graph Structure
               Learning and Next Interaction Prediction},
  booktitle = {{WWW} '22: The {ACM} Web Conference 2022, Virtual Event, Lyon, France,
               April 25 - 29, 2022},
  pages     = {2120--2130},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3485447.3512085},
  doi       = {10.1145/3485447.3512085},
  timestamp = {Tue, 26 Apr 2022 16:02:09 +0200},
  biburl    = {https://dblp.org/rec/conf/www/WeiBBW22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



