## Directory Structure

`lib` contains all of our core codebase.

`app` contains standalone applications that users can run.

## Training & Rendering

**Note.** All following commands should be ran within the `sdf-net` directory.

### Download sample data

To download a cool armadillo:

```
wget https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj -P data/
```

To download a cool matcap file:

```
wget https://raw.githubusercontent.com/nidorx/matcaps/master/1024/6E8C48_B8CDA7_344018_A8BC94.png -O data/matcap/green.png
```

### Training from scratch

```
python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/armadillo_normalized.obj \
    --raw-obj-path data/armadillo.obj \
    --epoch 250 \
    --exp-name armadillo
```

This will populate `_results` with TensorBoard logs.

### Rendering the trained model

If you set custom network parameters in training, you need to also reflect them for the renderer.

For example, if you set `--feature-dim 16` above, you need to set it here too.

```
python app/sdf_renderer.py \
    --net OctreeSDF \
    --num-lods 5 \
    --pretrained _results/models/armadillo.pth \
    --render-res 1280 720 \
    --shading-mode matcap \
    --lod 4
```

By default, this will populate `_results` with the rendered image.

If you want to export a `.npz` model which can be loaded into the C++ real-time renderer, add the argument 
`--export path/file.npz`. Note that the renderer only supports the base Neural LOD configuration
(the default parameters with `OctreeSDF`).

## Core Library Development Guide

To add new functionality, you will likely want to make edits to the files in `lib`. 

We try our best to keep our code modular, such that key components such as `trainer.py` and `renderer.py` 
need not be modified very frequently to add new functionalities.

To add a new network architecture for an example, you can simply add a new Python file in `lib/models` that
inherits from a base class of choice. You will probably only need to implement the `sdf` method which 
implements the forward pass, but you have the option to override other methods as needed if more custom
operations are needed. 

By default, the loss function used are defined in a CLI argument, which the code will automatically parse
and iterate through each loss function. The network architecture class is similarly defined in the CLI 
argument; simply use the exact class name, and don't forget to add a line in `__init__.py` to resolve the 
namespace.

## App Development Guide

To make apps that use the core library, add the `sdf-net` directory into the Python `sys.path`, so 
the modules can be loaded correctly. Then, you will likely want to inherit the same CLI parser defined
in `lib/options.py` to save time. You can then add a new argument group `app` to the parser to add custom
CLI arguments to be used in conjunction with the defaults. See `app/sdf_renderer.py` for an example.

Examples of things that are considered `apps` include, but are not limited to:

- visualizers
- training code
- downstream applications


