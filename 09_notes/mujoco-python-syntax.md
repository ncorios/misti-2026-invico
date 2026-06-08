# MUJOCO NOTES

## Two Main Objects
### *model* and *data*
*model:*
    model = mujoco.MjModel.from_xml_path("xml_path")

    model is the description. masses, geometry, timestep, etc. nothing changes while sim runs.
*data:*
    data = mujoco.MjData(model) 

    data is the state. positions, velocities, forces, contacts, sensor readings, etc.

    q: generalized coordinates.

    nq is length of qpos array
    nv is length of qvel array


    qpos: generalized positions. 
    array structure:
    indices 0-2: x,y,z for base
    indices 3-6: base orientation, stored as quaternions qw, qx, qy, qz. 
    indices 7-n are your joint angles.
    ex: indexing joint anglesthrough qpos is len(data.qpos[7:])

    qvel: generalized velocities
    shorter than qpos, bc quaternion is 4 for position, but you only need 3 angular velocites to describe rotation
    indices 0-2: v_x,v_y,v_z for base
    indices 3-5: ωx, ωy, ωz for base
    indices 6-n: joint velocities


    qacc: generalized accelerations
    same indexing as velocity. useful print for catching overshoot faster than lower order terms.

    qfrc_: family of generalized forces

    data.qfrc_bias: family of bias forces; gravituy, coriolis, centrifugal all stored here. apply qfrc_bias as your control torque for gravity compensation feedforward.

    data,qfrc_applied: a slot in data that starts empty. you can inject torques to simulate external forces acting on the plant. raw injection, doesnt get cleared by the engine. essentially a user force-injection buffer

    indices for these two:

    0-2: f_x, f_y, f_z for base
    3-5: tau_x, tau_y, tau_z for base rotation
    offset 6 for joint torques

    now what you actually write to.

    data.ctrl:
    an array of model.nu (number of actuators).
    with <positio> actuators, each ctrl value is a target angle, and MuJoCo curns its own PD controller internally.
    with <motor> actuators, each ctrl is a
    raw torque for the motors to apply.
    indices correspond to actuators.

## Loop Drivers"
    program has a rhythm: step, read compute, write, repeat. read after a step has occured. step should happen first.

    functions with mj_ prefixes are c functions that mutate data.

    mujuco.mj_step(model,data): advances the sim by one model.opt.timestep. it reads data.ctrl, computes every forces, solves for qacc, and integrates that forward into new qvel and qpos

    mujoco.mj_name2id(model, type, name): converts a string name from XML into the integer index. type: mjOBJ_JOINT, mjOBJ_BODY, mjOBJ_GEOM, mjOBJ_KEY.

    mj_id2name(model, type, id) is its reverse.

    mj_forward(model, data): runs the kinematics and dyanmics pipeline without integration. for outputs without running time forward.

    mj_reseData(model, data) resets the state to model defaults. mj_resetDataKeyFrame(model, data, key_id) resets to a stored keyframe.

    viewer

   with  mujoco.viewer.launch_passive(model,data) as viewer:
   viewer renders whats in data, doesnt run physics. call viewer.sync() once per iteration.


    timesteps:

    model.opt.timestep: physics timestep/
    control timestep: physics timestep * N
    N is how many times you step physics per control step. you call mj_step several times in a row before recomputing. 

    procedure:
    pick physics timestep.
    pick control frequency, 10-20x faster than fastes dynamics in the system.
    N = physics timestep / control freq (substep)
    control_dt = physics_timestep * N
    try to use N = 1 first, more to save compute









