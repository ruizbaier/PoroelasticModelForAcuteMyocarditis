# PoroelasticModelForAcuteMyocarditis

## What
Finite element simulation of immune system dynamics coupled with large-strain poromechanics. It relies on the [FEniCS](https://fenicsproject.org) package (tested with v.2019.1).

## Context 
The immune system dynamics are as follows 

![](https://github.com/ruizbaier/PoroelasticModelForAcuteMyocarditis/blob/main/scheme.png)

And the simulations produce the following type of solutions 

![](https://github.com/ruizbaier/PoroelasticModelForAcuteMyocarditis/blob/main/snapshots.png)


## How to cite
Further details on the model and discretisation can be found in the following two references [[1]](http://arxiv.org/abs/2111.04206) [[2]](https://www.frontiersin.org/articles/10.3389/fphys.2022.888515)
```
@rticle{Barnafi21,
  url     = {http://arxiv.org/abs/2111.04206},
  doi     = {2111.04206xxx},
  author  = {Barnafi, Nicolas and G\'omez-Vargas, Bryan  and Louren\c{c}o, Wesley de Jesus and 
              Reis, Ruy Freitas and Rocha, Bernardo Martins and Lobosco, Marcelo and 
              Ruiz-Baier, Ricardo and Weber dos Santos, Rodrigo},
  title   = {Mixed methods for large-strain poroelasticity/chemotaxis models 
              simulating the formation of myocardial oedema},
  year    = {2021},
  journal = {arXiv preprint}
}

@article{Lourenco22,
  doi = {10.3389/fphys.2022.888515},
  year    = {2022},
  volume  = {13}, 
  pages ={e888515(1--14)},
  author  = {Louren\c{c}o, Wesley de Jesus and Reis, Ruy Freitas and 
              Ruiz-Baier, Ricardo and Rocha, Bernardo Martins and 
              Weber dos Santos, Rodrigo and Lobosco, Marcelo},
  title   = {A poroelastic approach for modelling myocardial oedema 
              in acute myocarditis},
  journal = {Frontiers in Physiology}
}
```

