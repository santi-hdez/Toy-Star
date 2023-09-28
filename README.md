# Toy-Star

This procedure simulates a polytropic, 3D star model. We choose a sphere with radius R and mass content M. We place
particles with mass M/N randomly within this sphere and integrate the hydro equations using the SPH algorithm under
consideration of a damping force and a simplified gravitational term. 
 
If run with the default values, the following will be observed during the simulation:
     
   1. Since the initial particle distribution is not in hydrostatic equilibrium, the whole particle distribution will
      expand.
   2. The particles will execute a damped oscillation around the final equilibrium state due to the attracting
      gravitational term lambd*x and the linear damping term damp*v.

More information about the execution of this procedure can be found in the documentation of the code.
