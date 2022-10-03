# Feature Lines Extraction

FeatureLine is used to extract feature lines as per the paper (Gumhold, Stefan & Wang, Xinlong & MacLeod, Rob. (2001). Feature Extraction from Point Clouds. Proceedings of 10th international meshing roundtable. 2001. )

## install requirements
after creating the environment 

```console
foo@bar:~$ conda install --file requirements.txt
```
or 

```console
foo@bar:~$ pip3 install requirements.txt
```

## Usage

```python
from helper import FeatureLines

# returns 'Object after the downsampling specified'
Obj1 = FeatureLines("data/group19/fragment.ply",voxel_size=0.1)

# calculates all penalty functions after creating the graph where N = 16 (nearest neighbor)
Obj1.init(16)

```
## Penalties calculated
w_k : curvature estimate for each node (one value per node)

w_cr_v : vector valued penalty on how well the eigenvalues fit to the crease ( 3D vector value per node )

w_b1 : vector valued border penalty function ( 3D vector value per node )

w_b2 : probability of being border point ( one value per node )

w_co : The corner penalty function ( one value per node )

## Show Results

```python
data = Obj1.w_k
Obj1.show_heat(data)
```
![alt text](https://github.com/RePAIRProject/AAFR/blob/master/Trials/w_k.JPG)
```python
data = Obj1.w_cr_v 
Obj1.show_heat(data)
```
![alt text](https://github.com/RePAIRProject/AAFR/blob/master/Trials/w_cr.JPG)
```python
data = Obj1.w_b1
Obj1.show_heat(data)
```
![alt text](https://github.com/RePAIRProject/AAFR/blob/master/Trials/w_b1.JPG)
```python
data = Obj1.w_b2
Obj1.show_heat(data)
```
![alt text](https://github.com/RePAIRProject/AAFR/blob/master/Trials/w_b2.JPG)
```python
data = Obj1.w_co
Obj1.show_heat(data)
```
![alt text](https://github.com/RePAIRProject/AAFR/blob/master/Trials/w_co.JPG)


# Experiment module
## Default Defined patterns Creation

```python
from runner import experiment

## load the config file for the expirments 
my_expirments = experiment("conf1.yaml")

## run all expirments in parallel depending on the number of cores avaliable 
my_expirments.run()
## show results
my_expirments.results
```
