#Imports:
import numpy as np
import matplotlib.pyplot as plt

# definitions
def main(
    grid_width,   # number of x-steps
    grid_height,   # number of y-steps

    num_neurons,
    num_amacrine_repellents,
    num_muller_repellents,
    num_attractors,

    sigma_amacrine,
    sigma_muller,
    sigma_attractor,

    amacrine_strength,
    muller_strength,
    attractor_strength,

    max_steps,
    include_plot = True
):
    """
    Main function runs a simulation
    """
    step_size = 1.0 / max(grid_width, grid_height)

    # Grid setup
    x = np.linspace(0, 1, grid_width)
    y = np.linspace(0, 1, grid_height)
    x, y = np.meshgrid(x, y)

    # generate field (return total_field)
    total_field, amacrine_positions, repellent_positions_muller, attractor_positions = generate_field(
        num_amacrine_repellents=num_amacrine_repellents,
        num_muller_repellents=num_muller_repellents,
        num_attractors=num_attractors,

        sigma_amacrine=sigma_amacrine,
        sigma_muller=sigma_muller,
        sigma_attractor=sigma_attractor,
        
        amacrine_strength=amacrine_strength,
        muller_strength=muller_strength,
        attractor_strength=attractor_strength,

        x=x,
        y=y
    )
    
    # compute neuron trajectories and return ND array with them + a list of indexes for color coding
    all_trajectories = compute_neuron_trajectory(
        total_field=total_field,
        grid_width=grid_width,
        grid_height=grid_height,
        num_neurons=num_neurons,
        step_size=step_size,
        max_steps=max_steps
    )
    
    # Generate a plot
    if include_plot:
        # use AXIS where you would use PLT in original code
        fig, axis = plot_field(total_field=total_field, grid_height=grid_height, grid_width=grid_width)
        axis = plot_neurons(all_trajectories=all_trajectories, axis=axis, grid_width=grid_width, grid_height=grid_height, num_neurons=num_neurons)
        axis = plot_atr_rep(
            axis=axis,
            grid_width=grid_width,
            grid_height=grid_height,
            amacrine_positions=amacrine_positions,
            repellent_positions_muller=repellent_positions_muller,
            attractor_positions=attractor_positions
        )
        
        plt.title("Axon Guidance: Slit Repellents, Müller Glia, and Attractors")
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

def generate_field(
    num_amacrine_repellents,
    num_muller_repellents,
    num_attractors,

    sigma_amacrine,
    sigma_muller,
    sigma_attractor,
    
    amacrine_strength,
    muller_strength,
    attractor_strength,

    x,
    y
):
    # --- Generate field components separately ---

    # 1. Amacrine repellent field
    amacrine_positions = [
        (np.random.uniform(0.0, 0.45) if np.random.rand() < 0.5 else np.random.uniform(0.55, 1.0), 
        np.random.uniform(0.0, 0.2)) 
        for _ in range(num_amacrine_repellents)
    ]
    amacrine_field = np.zeros_like(x)

    for rx, ry in amacrine_positions:
        amacrine_field += amacrine_strength * np.exp(-((x - rx)**2 + (y - ry)**2) / (2 * sigma_amacrine**2))

    # 2. Müller glia repellent field
    repellent_positions_muller = [
        (np.random.uniform(0, 1), np.random.uniform(0.8, 1)) for _ in range(num_muller_repellents)
    ]
    muller_field = np.zeros_like(x)
    for rx, ry in repellent_positions_muller:
        muller_field += muller_strength * np.exp(-((x - rx)**2 + (y - ry)**2) / (2 * sigma_muller**2))

    # 3. Attractor field (optic disc)
    attractor_positions = [
        (np.random.uniform(0.475, 0.525), np.random.uniform(0.0, 0.2)) for _ in range(num_attractors)
    ]
    attractor_field = np.zeros_like(x)
    for ax, ay in attractor_positions:
        attractor_field -= attractor_strength * np.exp(-((x - ax)**2 + (y - ay)**2) / (2 * sigma_attractor**2))

    # --- Combine all fields ---
    total_field = amacrine_field + muller_field + attractor_field

    return total_field, amacrine_positions, repellent_positions_muller, attractor_positions

def compute_neuron_trajectory(
        total_field,
        grid_width,
        grid_height,
        num_neurons,
        step_size,
        max_steps
    ):
    # --- Compute field gradient ---
    grad_y, grad_x = np.gradient(total_field)
    print(grad_y, grad_x)

    # --- Neuron starting positions ---
    starting_positions = [
        (np.random.uniform(0, 1), np.random.uniform(0.25, 0.3)) for _ in range(num_neurons)
    ]

        # --- Simulate each neuron and save trajectories of---

    all_trajectories = []
    for idx, start_pos in enumerate(starting_positions):
        fpos = np.array(start_pos, dtype=float)
        trajectory = []

        for _ in range(max_steps):
            trajectory.append(fpos.copy())
            i = int(fpos[1] * (grid_height - 1))
            j = int(fpos[0] * (grid_width - 1))
            if i < 0 or i >= grid_height or j < 0 or j >= grid_width:
                break
            direction = np.array([grad_x[i, j], grad_y[i, j]])
            norm = np.linalg.norm(direction)
            if norm == 0: # If gradient at a point is 0 the neuron should take a random step to mimic real cell behaviour
                random_direction = np.random.randn(2)
                random_direction /= np.linalg.norm(random_direction)
                fpos += step_size * random_direction
            else:
                fpos -= step_size * direction / norm
            
        trajectory = np.array(trajectory)
        all_trajectories.append((idx, trajectory))

    return all_trajectories

def plot_field(total_field, grid_height, grid_width):
    """
    Plots the total field     
    """
    fig, axis = plt.subplots()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    cax = axis.imshow(total_field, cmap='viridis', origin='lower', extent=[0, grid_width, 0, grid_height])
    fig.colorbar(cax, label='Field Intensity')

    return fig, axis

def plot_neurons(all_trajectories, axis, grid_width, grid_height, num_neurons):

    colors = plt.cm.plasma(np.linspace(0, 1, num_neurons))
    for idx, trajectory in all_trajectories:
        axis.plot(trajectory[:, 0] * grid_width, trajectory[:, 1] * grid_height, color=colors[idx], linewidth=2)
    
    return axis
    
def plot_atr_rep(axis, grid_width, grid_height, amacrine_positions, repellent_positions_muller, attractor_positions):
    # --- Visual markers ---
    for rx, ry in amacrine_positions:
        axis.scatter(rx * grid_width, ry * grid_height, color='white', edgecolor='black', label='Slit Repellent' if rx == amacrine_positions[0][0] else "")

    for rx, ry in repellent_positions_muller:
        axis.scatter(rx * grid_width, ry * grid_height, color='blue', edgecolor='black', s=10, label='Müller Glia' if rx == repellent_positions_muller[0][0] else "")

    for ax, ay in attractor_positions:
        axis.scatter(ax * grid_width, ay * grid_height, color='red', edgecolor='black', label='Attractor' if ax == attractor_positions[0][0] else "")
    
    return axis


if __name__ == "__main__": # this if clause stops the code from being executed when imported in another file


    main(
        grid_width = 160,   # number of x-steps
        grid_height = 80,   # number of y-steps
        num_neurons = 50,
        num_amacrine_repellents = 50,
        num_muller_repellents = 50,
        num_attractors = 10,
        sigma_amacrine = 0.2,
        sigma_muller = 0.2,
        sigma_attractor = 0.3,
        amacrine_strength = 1.0,
        muller_strength = 1.0,
        attractor_strength = 5.0,
        max_steps = 100
    )

