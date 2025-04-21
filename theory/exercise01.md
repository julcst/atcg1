# Assignment 3) Radiance and Irradiance

Given the radiance of the sun $L_s = 20.045 \frac{MW}{m^2 \cdot sr}$ and the direction $v$ having an angle of 45◦ to the table, calculate the irradiance at the table and the total radiant power incident on the table plate. For this, assume that the radiance is constant.

## Solution

#### Irradiance

The irradiance $E$ is given by the formula:

$$
E = \frac{\mathrm{d}\Phi}{\mathrm{d}A} = \int_\Omega L_s \cos\varepsilon \; \mathrm{d}\Omega = 20.045 \frac{MW}{m^2 \cdot sr} \cdot \cos(45^\circ) \approx 14.174 \frac{MW}{m^2}
$$

#### Total Radiant Power

The total radiant power $\Phi$ incident on the table plate is given by the formula:

$$
\Phi = \int_A E \; \mathrm{d}A = E \cdot A \approx 14.174 \frac{MW}{m^2} \cdot 0.8 m \cdot 0.8 m \approx 907133.147 MW
$$