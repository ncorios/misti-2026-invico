# MUJOCO NOTES

## Two Main Objects
### *model* and *object*
*model:*
    model = mujoco.MjModel.from_xml_path("xml_path")
    model is the description. masses, geometry, timestep, etc. nothing changes while sim runs.
*data:*
    data = mujoco.MjData(model) 
    data is the state. positions, velocities, forces, contacts, sensor readings, etc.
    
