# Workflows
1. Download the SMPLX models from the official website, and put it under the folder (according to the path in code)
2. Open your 3D character OBJ file in Blender and then select the markers on your OBJ mesh (must follow the selection scheme here! Please check "how to label markers" folder)
3. save it as json file with key-value pair (eg. position: coordinate)
4. replace the file paths
5. python ShapeFitter.py
(If you want, you can decide your own markers, but you also need to change the markers for the SMPLX model)

# Some Results
~ Adjust the parameters by yourself, eg.loops, regularization, ...
