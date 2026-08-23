"""Streamlit web interface with real-time animated visualization."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from dataclasses import replace
import tempfile
import os

# Import simulation components
from config import DEFAULT_CONFIG, EXPERIMENT_PRESETS, SimulationConfig
from src.simulation.world import World
from src.simulation.species import PREY, PREDATOR

# Set Streamlit page config
st.set_page_config(
    page_title="Emergent Ecosystems",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { max-width: 1400px; }
    .stMetric { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Emergent Ecosystems: Evolved Neural NPCs vs. Hand-Coded Game AI")
st.markdown("""
A 2D artificial-life simulation where autonomous prey and predator NPCs compete using:
- **Neural Networks** (evolved via neuroevolution)
- **Utility-based AI** (hand-coded decision rules)  
- **Finite-State Machines** (baseline controller)
""")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Decision mode
    decision_mode = st.selectbox(
        "Decision Policy",
        options=["neural", "utility", "fsm"],
        help="Switch between evolved neural networks, utility-based AI, or FSM baseline"
    )
    
    # Scenario
    scenario = st.selectbox(
        "Scenario",
        options=["baseline", "resource_scarcity", "predator_pressure"],
        help="Different environmental conditions"
    )
    
    # Simulation parameters
    st.subheader("Initial Population")
    initial_prey = st.slider("Prey", 10, 100, EXPERIMENT_PRESETS[scenario].initial_prey)
    initial_predators = st.slider("Predators", 1, 30, EXPERIMENT_PRESETS[scenario].initial_predators)
    initial_resources = st.slider("Resources", 10, 200, EXPERIMENT_PRESETS[scenario].initial_resources)
    
    st.subheader("Simulation Speed")
    ticks = st.slider("Ticks to run", 100, 2000, 300, step=100)
    updates_per_frame = st.slider("Updates per frame", 1, 5, 1)
    
    seed_val = st.number_input("Random seed", value=42, min_value=0)
    
    run_simulation = st.button("▶️ Run Simulation", use_container_width=True)

# Main area
if run_simulation:
    # Create config
    config = EXPERIMENT_PRESETS[scenario]
    config = replace(
        config,
        decision_mode=decision_mode,
        initial_prey=initial_prey,
        initial_predators=initial_predators,
        initial_resources=initial_resources,
        seed=seed_val,
        updates_per_frame=updates_per_frame,
        screen_width=800,
        screen_height=600,
        world_width=800,
        world_height=600,
    )
    
    # Run simulation
    with st.spinner(f"Running {ticks} ticks with {decision_mode} policy..."):
        try:
            world = World(config)
            world.populate_initial()
            
            # Storage for frames and metrics
            frames = []
            metrics_history = {
                "tick": [],
                "prey_count": [],
                "predator_count": [],
                "resource_count": [],
                "avg_prey_speed": [],
                "avg_predator_speed": [],
            }
            
            # Run ticks and capture frames every N ticks
            progress_bar = st.progress(0)
            frame_interval = max(1, ticks // 60)  # Capture ~60 frames for smooth animation
            
            for tick in range(ticks):
                world.update(config.dt)
                
                # Capture frame
                if tick % frame_interval == 0 or tick == ticks - 1:
                    frame_data = {
                        "tick": tick,
                        "agents": [(a.position, a.species) for a in world.agents if a.alive],
                        "resources": [r.position for r in world.resources],
                    }
                    frames.append(frame_data)
                
                # Collect metrics
                collection_interval = max(1, ticks // 50)
                if tick % collection_interval == 0 or tick == ticks - 1:
                    latest = world.metrics.latest_record()
                    if latest:
                        metrics_history["tick"].append(latest["tick"])
                        metrics_history["prey_count"].append(latest["prey_population"])
                        metrics_history["predator_count"].append(latest["predator_population"])
                        metrics_history["resource_count"].append(latest["resource_count"])
                        metrics_history["avg_prey_speed"].append(latest["avg_prey_speed"])
                        metrics_history["avg_predator_speed"].append(latest["avg_predator_speed"])
                
                progress_bar.progress(min((tick + 1) / ticks, 1.0))
            
            # Display results
            st.success(f"✅ Simulation complete!")
            
            # Create two columns: animation and metrics
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🎮 World Visualization")
                
                # Create animation with matplotlib
                fig, ax = plt.subplots(figsize=(10, 7.5), dpi=80)
                fig.patch.set_facecolor('#131717')
                
                def animate_frame(frame_idx):
                    ax.clear()
                    frame = frames[frame_idx]
                    
                    # Background
                    ax.set_facecolor('#131717')
                    ax.set_xlim(0, world.width)
                    ax.set_ylim(0, world.height)
                    ax.set_aspect('equal')
                    ax.invert_yaxis()
                    
                    # Draw resources (green)
                    for pos in frame["resources"]:
                        circle = Circle(pos, 4, color='#58D26A', alpha=0.7)
                        ax.add_patch(circle)
                    
                    # Draw agents
                    for pos, species in frame["agents"]:
                        if species == PREY:
                            color = '#46A5FF'  # Blue
                            size = 40
                        else:
                            color = '#EB534C'  # Red
                            size = 50
                        ax.scatter(pos[0], pos[1], s=size, c=color, alpha=0.8, edgecolors='white', linewidth=0.5)
                    
                    # Title with tick count
                    ax.set_title(f"Tick: {frame['tick']} | Policy: {decision_mode} | Scenario: {scenario.replace('_', ' ').title()}", 
                                fontsize=12, color='white', pad=10)
                    ax.set_xlabel("X", color='white')
                    ax.set_ylabel("Y", color='white')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.15, color='white')
                    
                    # Legend
                    prey_count = sum(1 for _, s in frame["agents"] if s == PREY)
                    pred_count = sum(1 for _, s in frame["agents"] if s == PREDATOR)
                    res_count = len(frame["resources"])
                    ax.text(0.02, 0.98, f"🔵 Prey: {prey_count}  🔴 Predators: {pred_count}  🟢 Resources: {res_count}", 
                           transform=ax.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='#0C1011', alpha=0.8, edgecolor='white'),
                           color='white')
                
                # Create animation object
                anim = animation.FuncAnimation(fig, animate_frame, frames=len(frames), 
                                             interval=50, repeat=True, blit=False)
                
                # Save animation to temporary file
                with tempfile.TemporaryDirectory() as tmpdir:
                    anim_path = os.path.join(tmpdir, "animation.gif")
                    anim.save(anim_path, writer='pillow', fps=20, dpi=80)
                    
                    # Read and display
                    with open(anim_path, 'rb') as f:
                        st.image(f.read(), use_container_width=True)
                
                plt.close(fig)
                st.caption(f"Animation ({len(frames)} frames @ 20 fps)")
            
            with col2:
                st.subheader("📊 Real-time Metrics")
                
                # Final stats
                st.metric("Final Prey", int(metrics_history["prey_count"][-1]) if metrics_history["prey_count"] else 0)
                st.metric("Final Predators", int(metrics_history["predator_count"][-1]) if metrics_history["predator_count"] else 0)
                st.metric("Final Resources", int(metrics_history["resource_count"][-1]) if metrics_history["resource_count"] else 0)
            
            # Full-width charts
            st.subheader("📈 Population Over Time")
            
            if metrics_history["tick"]:
                df = pd.DataFrame(metrics_history)
                
                # Population chart
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df["tick"], df["prey_count"], label="Prey", color="#46A5FF", linewidth=2.5, marker='o', markersize=3)
                ax.plot(df["tick"], df["predator_count"], label="Predators", color="#EB534C", linewidth=2.5, marker='s', markersize=3)
                ax.plot(df["tick"], df["resource_count"], label="Resources", color="#58D26A", linewidth=2.5, marker='^', markersize=3)
                ax.set_xlabel("Tick", fontsize=11)
                ax.set_ylabel("Count", fontsize=11)
                ax.set_title(f"Population Dynamics - {scenario.replace('_', ' ').title()} ({decision_mode})")
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Speed chart
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.plot(df["tick"], df["avg_prey_speed"], label="Avg Prey Speed", color="#46A5FF", linewidth=2.5, marker='o', markersize=3)
                ax.plot(df["tick"], df["avg_predator_speed"], label="Avg Predator Speed", color="#EB534C", linewidth=2.5, marker='s', markersize=3)
                ax.set_xlabel("Tick", fontsize=11)
                ax.set_ylabel("Speed", fontsize=11)
                ax.set_title("Average Agent Speed Over Time")
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download metrics as CSV",
                    data=csv,
                    file_name=f"emergent_ecosystem_{scenario}_{decision_mode}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.info("Try running locally for better debugging: `streamlit run streamlit_app.py`")

else:
    # Welcome screen
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Choose a policy** (Neural, Utility, or FSM) from the sidebar
        2. **Select a scenario** (baseline, scarcity, or predator pressure)
        3. **Adjust initial populations** as desired
        4. **Click "Run Simulation"** to watch the ecosystem evolve in real-time!
        
        ### What you'll see:
        - 🎮 **Live animated world visualization** with prey (blue), predators (red), and resources (green)
        - 📊 **Real-time population charts** tracking all three species
        - 📉 **Speed metrics** showing average agent trait evolution
        
        ### What are the policies?
        - **Neural**: Agents use evolved neural networks (neuroevolution, no backprop)
        - **Utility**: Hand-coded decision rules with scoring
        - **FSM**: Finite-state machine baseline controller
        
        All three policies share the same perception, actions, and physiology — **only the brain differs**.
        """)
    
    with col2:
        st.subheader("📚 Resources")
        st.markdown("""
        - [GitHub Repo](https://github.com/theapuhasan/emergent-ecosystem)
        - [Desktop App](https://github.com/theapuhasan/emergent-ecosystem#running-the-simulation)
        - [Local Setup](https://github.com/theapuhasan/emergent-ecosystem#installation)
        """)
    
    st.info("""
    **💡 Tip**: Start with the "baseline" scenario on "utility" mode to see stable populations, 
    then experiment with "neural" mode to see how evolved controllers perform!
    """)

st.markdown("---")
st.caption("Emergent Ecosystems © 2024 | A study in artificial life and game AI | 🎮 Interactive Web Visualization")
