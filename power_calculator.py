"""
Power Calculator for Gas Mixture Heating
Estimates power required to heat a gas mixture with CO2 conversion reaction
"""

import tkinter as tk
from tkinter import ttk
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import warnings
from databaseHandler import Handler

# Suppress matplotlib tight_layout warnings (harmless but noisy)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Reference state for all enthalpy calculations
# All gas enthalpies are calculated relative to this state for consistency
REF_TEMP_K = 423.15  # 150°C in Kelvin
REF_PRESSURE_PA = 101325  # 1 atm in Pa
R = 8.314  # Gas constant J/(mol·K)

# Heat of formation at STP (kJ/mol) - from NIST
H_FORMATION = {
    'CO2': -393.5,  # kJ/mol
    'CO': -110.5,   # kJ/mol
    'H2': 0.0,      # kJ/mol (reference)
    'H2O': -241.8,  # kJ/mol (for water vapor)
    'N2': 0.0       # kJ/mol (reference)
}

# Molar masses (g/mol)
MOLAR_MASS = {
    'CO2': 44.01,
    'CO': 28.01,
    'H2': 2.016,
    'H2O': 18.015,
    'N2': 28.014
}

class ThermodynamicCalculator:
    """Handles thermodynamic calculations using CoolProp/NIST data"""
    
    def __init__(self):
        # CoolProp fluid names
        self.fluids = {
            'CO2': 'CarbonDioxide',
            'CO': 'CarbonMonoxide',
            'H2': 'Hydrogen',
            'H2O': 'Water',
            'N2': 'Nitrogen'
        }
        
        # Cache reference state enthalpy values for consistent reference state
        self._h_ref_cache = {}
        self._initialize_reference()
    
    def _initialize_reference(self):
        """Initialize reference state enthalpies for all gases at 150°C, 1 atm"""
        for gas, fluid in self.fluids.items():
            try:
                # Get absolute enthalpy at reference state (150°C, 1 atm) from CoolProp
                h_ref_abs = CP.PropsSI('H', 'T', REF_TEMP_K, 'P', REF_PRESSURE_PA, fluid)
                self._h_ref_cache[gas] = h_ref_abs
            except Exception as e:
                print(f"Warning: Could not get reference enthalpy for {gas}: {e}")
                self._h_ref_cache[gas] = 0.0
    
    def get_enthalpy(self, gas, temperature_k, pressure_pa=REF_PRESSURE_PA):
        """
        Get specific enthalpy (J/kg) at given temperature relative to reference state
        All gases use the same reference state: 150°C, 1 atm
        """
        fluid = self.fluids[gas]
        try:
            # Get absolute enthalpy at desired temperature from CoolProp
            h_t_abs = CP.PropsSI('H', 'T', temperature_k, 'P', pressure_pa, fluid)
            
            # Get reference state enthalpy (cached)
            h_ref_abs = self._h_ref_cache.get(gas, 0.0)
            if h_ref_abs == 0.0:
                # Fallback: calculate reference enthalpy if not cached
                h_ref_abs = CP.PropsSI('H', 'T', REF_TEMP_K, 'P', REF_PRESSURE_PA, fluid)
                self._h_ref_cache[gas] = h_ref_abs
            
            # Return enthalpy relative to reference state (consistent reference state)
            return h_t_abs - h_ref_abs
        except Exception as e:
            print(f"Error getting enthalpy for {gas} at {temperature_k}K: {e}")
            return 0.0
    
    def get_molar_enthalpy(self, gas, temperature_k, pressure_pa=REF_PRESSURE_PA):
        """
        Get molar enthalpy (J/mol) at given temperature relative to reference state
        All gases use the same reference state: 150°C, 1 atm
        """
        h_specific = self.get_enthalpy(gas, temperature_k, pressure_pa)
        molar_mass = MOLAR_MASS[gas] / 1000  # Convert to kg/mol
        return h_specific * molar_mass
    
    def get_enthalpy_at_reference(self, gas):
        """
        Get absolute enthalpy (J/kg) at reference state (150°C, 1 atm) from CoolProp
        Used for calculating heat of formation differences
        """
        fluid = self.fluids[gas]
        try:
            return CP.PropsSI('H', 'T', REF_TEMP_K, 'P', REF_PRESSURE_PA, fluid)
        except Exception as e:
            print(f"Error getting reference enthalpy for {gas}: {e}")
            return 0.0
    
    def calculate_reaction_enthalpy(self, co2_moles, h2_moles, temperature_k):
        """
        Calculate enthalpy change for reaction at given temperature:
        CO2 + H2 -> CO + H2O
        
        Uses CoolProp for all enthalpy calculations with consistent reference state (150°C, 1 atm).
        Accounts for:
        1. Heat of formation at reference state (calculated from CoolProp enthalpies)
        2. Sensible heat difference between products and reactants at reaction temperature
        
        Returns enthalpy change in J
        """
        # Calculate heat of formation from CoolProp enthalpies at reference state
        # Get absolute enthalpies at reference state (150°C, 1 atm) (J/kg)
        h_co2_ref = self.get_enthalpy_at_reference('CO2')
        h_h2_ref = self.get_enthalpy_at_reference('H2')
        h_co_ref = self.get_enthalpy_at_reference('CO')
        h_h2o_ref = self.get_enthalpy_at_reference('H2O')
        
        # Convert to molar enthalpies (J/mol)
        h_co2_ref_mol = h_co2_ref * (MOLAR_MASS['CO2'] / 1000)
        h_h2_ref_mol = h_h2_ref * (MOLAR_MASS['H2'] / 1000)
        h_co_ref_mol = h_co_ref * (MOLAR_MASS['CO'] / 1000)
        h_h2o_ref_mol = h_h2o_ref * (MOLAR_MASS['H2O'] / 1000)
        
        # Heat of formation from NIST (relative to elements at standard conditions)
        # We use NIST standard heat of formation values as they're more accurate
        # for chemical reactions, but we verify with CoolProp sensible heat
        delta_h_form_nist = (H_FORMATION['CO'] + H_FORMATION['H2O'] - 
                            H_FORMATION['CO2'] - H_FORMATION['H2']) * 1000  # J/mol
        
        # Sensible heat at reaction temperature (relative to reference state) - all from CoolProp
        # Products: CO + H2O
        h_co = self.get_molar_enthalpy('CO', temperature_k)
        h_h2o = self.get_molar_enthalpy('H2O', temperature_k)
        
        # Reactants: CO2 + H2 (already at reaction temperature)
        h_co2 = self.get_molar_enthalpy('CO2', temperature_k)
        h_h2 = self.get_molar_enthalpy('H2', temperature_k)
        
        # Total enthalpy change per mole of reaction at temperature
        # = Sensible heat difference (from CoolProp) + Heat of formation (NIST standard)
        # All sensible heat calculations use consistent reference state (150°C, 1 atm)
        delta_h_reaction = (h_co + h_h2o - h_co2 - h_h2) + delta_h_form_nist
        
        # Total enthalpy change for the reaction
        n_reaction = min(co2_moles, h2_moles)
        return delta_h_reaction * n_reaction


class PowerCalculator:
    """Main calculator for power requirements"""
    
    def __init__(self):
        self.thermo = ThermodynamicCalculator()
    
    def slpm_to_mol_per_sec(self, slpm):
        """Convert SLPM (Standard Liters Per Minute) to mol/s at STP"""
        # At STP: 1 mol = 22.414 L
        liters_per_sec = slpm / 60.0
        mol_per_sec = liters_per_sec / 22.414
        return mol_per_sec
    
    def calculate_power(self, inlet_temp_c, outlet_temp_c, 
                       co2_slpm, h2_slpm, n2_slpm, co2_conversion_pct):
        """
        Calculate power required to heat gas mixture
        
        Parameters:
        - inlet_temp_c: Inlet temperature in °C
        - outlet_temp_c: Outlet temperature in °C
        - co2_slpm, h2_slpm, n2_slpm: Volumetric flow rates in SLPM
        - co2_conversion_pct: Percentage of CO2 converted (0-100)
        """
        # Convert temperatures to Kelvin
        inlet_temp_k = inlet_temp_c + 273.15
        outlet_temp_k = outlet_temp_c + 273.15
        
        # Convert SLPM to mol/s
        n_co2 = self.slpm_to_mol_per_sec(co2_slpm)
        n_h2 = self.slpm_to_mol_per_sec(h2_slpm)
        n_n2 = self.slpm_to_mol_per_sec(n2_slpm)
        
        # Calculate moles converted
        n_co2_converted = n_co2 * (co2_conversion_pct / 100.0)
        n_h2_converted = min(n_co2_converted, n_h2)  # Same number of moles, but limited by available H2
        
        # Power for heating inlet gases to outlet temperature
        power_heating = 0.0
        
        # Heat CO2 (unconverted portion)
        n_co2_unconverted = n_co2 - n_co2_converted
        if n_co2_unconverted > 0:
            h_co2_in = self.thermo.get_molar_enthalpy('CO2', inlet_temp_k)
            h_co2_out = self.thermo.get_molar_enthalpy('CO2', outlet_temp_k)
            power_heating += n_co2_unconverted * (h_co2_out - h_co2_in)
        
        # Heat H2 (unconverted portion)
        n_h2_unconverted = n_h2 - n_h2_converted
        if n_h2_unconverted > 0:
            h_h2_in = self.thermo.get_molar_enthalpy('H2', inlet_temp_k)
            h_h2_out = self.thermo.get_molar_enthalpy('H2', outlet_temp_k)
            power_heating += n_h2_unconverted * (h_h2_out - h_h2_in)
        
        # Heat N2
        if n_n2 > 0:
            h_n2_in = self.thermo.get_molar_enthalpy('N2', inlet_temp_k)
            h_n2_out = self.thermo.get_molar_enthalpy('N2', outlet_temp_k)
            power_heating += n_n2 * (h_n2_out - h_n2_in)
        
        # Heat converted CO2 and H2 to outlet temperature, then account for reaction
        # The reaction occurs at outlet temperature: CO2 + H2 -> CO + H2O
        if n_co2_converted > 0 and n_h2_converted > 0:
            # Step 1: Heat CO2 from inlet to outlet temperature
            h_co2_in = self.thermo.get_molar_enthalpy('CO2', inlet_temp_k)
            h_co2_out = self.thermo.get_molar_enthalpy('CO2', outlet_temp_k)
            power_heating += n_co2_converted * (h_co2_out - h_co2_in)
            
            # Step 2: Heat H2 from inlet to outlet temperature
            h_h2_in = self.thermo.get_molar_enthalpy('H2', inlet_temp_k)
            h_h2_out = self.thermo.get_molar_enthalpy('H2', outlet_temp_k)
            power_heating += n_h2_converted * (h_h2_out - h_h2_in)
            
            # Step 3: Reaction enthalpy at outlet temperature
            # This accounts for heat of formation and sensible heat differences
            reaction_enthalpy = self.thermo.calculate_reaction_enthalpy(
                n_co2_converted, n_h2_converted, outlet_temp_k)
            power_heating += reaction_enthalpy
            
            # Note: Products (CO and H2O) are already at outlet temperature after reaction
        
        return power_heating  # Power in Watts


class PowerCalculatorGUI:
    """GUI application for power calculator"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Syngas Pilot Power Estimator")
        self.root.geometry("1200x1200")
        self.root.minsize(1000, 1000)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabelFrame.Label', font=("Consolas", 10, "bold"))
        self.style.configure('TLabel', font=("Consolas", 9))
        self.style.configure('TButton', font=("Consolas", 10))
        self.style.configure('TCheckbutton', font=("Consolas", 9))
        self.style.configure('TRadiobutton', font=("Consolas", 9))
        
        self.calculator = PowerCalculator()
        
        # Database handler
        try:
            self.db_handler = Handler()
            self.tag_id_dict = self.db_handler.getTagIDDictionary()
            self.db_available = True
        except Exception as e:
            print(f"Database not available: {e}")
            self.db_handler = None
            self.tag_id_dict = {}
            self.db_available = False
        
        # Data storage for plotting (unlimited history, 1 point per second)
        # power_history: (datetime, base_power_without_heat_loss, total_power, power_plus_25)
        self.power_history = deque()  
        self.temp_history = deque()   # [(datetime, side1_temp, side2_temp), ...]
        self.realtime_power_history = deque()  # [(datetime, realtime_power), ...]
        self.co2_flow_history = deque()  # [(datetime, co2_lpm), ...] for flow conversion tab
        self.n2_flow_history = deque()  # [(datetime, n2_lpm), ...] for flow conversion tab
        self.h2_flow_history = deque()  # [(datetime, h2_lpm), ...] for flow conversion tab
        
        # Live update control
        self.live_update_active = False
        self.update_thread = None
        self.update_lock = threading.Lock()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Power Calculator
        power_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(power_tab, text="Power Calculator")
        
        # Tab 2: Flow Rate Conversion
        conversion_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(conversion_tab, text="Flow Rate Conversion")
        
        # Create power calculator widgets in first tab
        self.create_power_calculator_tab(power_tab)
        
        # Create flow rate conversion widgets in second tab
        self.create_flow_conversion_tab(conversion_tab)
    
    def create_power_calculator_tab(self, parent):
        """Create widgets for the power calculator tab"""
        # Main frame
        main_frame = ttk.Frame(parent)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = tk.Label(main_frame, text="Syngas Pilot Power Estimator", 
                              font=("Consolas", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=15)
        
        # Left panel for inputs (fixed width)
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for plots (expands to fill)
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Temperature inputs
        temp_frame = ttk.LabelFrame(left_panel, text="", padding="10")
        temp_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        temp_frame.grid_columnconfigure(0, weight=1)
        temp_frame.grid_columnconfigure(1, weight=1)
        
        # Center the label frame title by creating a centered label
        temp_title = tk.Label(temp_frame, text="Temperature", 
                             font=("Consolas", 10, "bold"))
        temp_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Database mode toggle
        self.use_db_temps = tk.BooleanVar(value=False)
        db_check = tk.Checkbutton(temp_frame, text="Realtime",
                                 variable=self.use_db_temps,
                                 font=("Consolas", 9),
                                 command=self.toggle_db_mode)
        db_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        # HMR Side 1 Temp
        side1_label = tk.Label(temp_frame, text="HMR Side 1 Temp °C:", 
                                font=("Consolas", 9, "bold"))
        side1_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.side1_temp = tk.StringVar(value="25")
        self.side1_entry = tk.Entry(temp_frame, textvariable=self.side1_temp, width=15,
                              font=("Consolas", 9))
        self.side1_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        # HMR Side 1 Temp Tag 1
        side1_tag1_label = tk.Label(temp_frame, text="Side 1 Tag 1:", 
                                   font=("Consolas", 9, "bold"))
        side1_tag1_label.grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.side1_tag1 = tk.StringVar(value="ag")
        self.side1_tag1_entry = tk.Entry(temp_frame, textvariable=self.side1_tag1, width=15,
                                         font=("Consolas", 9), state='disabled')
        self.side1_tag1_entry.grid(row=3, column=1, sticky=tk.E, pady=5, padx=5)
        
        # HMR Side 1 Temp Tag 2
        side1_tag2_label = tk.Label(temp_frame, text="Side 1 Tag 2 (optional):", 
                                   font=("Consolas", 9, "bold"))
        side1_tag2_label.grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.side1_tag2 = tk.StringVar(value="aj")
        self.side1_tag2_entry = tk.Entry(temp_frame, textvariable=self.side1_tag2, width=15,
                                         font=("Consolas", 9), state='disabled')
        self.side1_tag2_entry.grid(row=4, column=1, sticky=tk.E, pady=5, padx=5)
        
        # HMR Side 2 Temp
        side2_label = tk.Label(temp_frame, text="HMR Side 2 Temp °C:", 
                               font=("Consolas", 9, "bold"))
        side2_label.grid(row=5, column=0, sticky=tk.W, pady=5, padx=5)
        self.side2_temp = tk.StringVar(value="800")
        self.side2_entry = tk.Entry(temp_frame, textvariable=self.side2_temp, width=15,
                               font=("Consolas", 9))
        self.side2_entry.grid(row=5, column=1, sticky=tk.E, pady=5, padx=5)
        
        # HMR Side 2 Temp Tag 1
        side2_tag1_label = tk.Label(temp_frame, text="Side 2 Tag 1:", 
                                     font=("Consolas", 9, "bold"))
        side2_tag1_label.grid(row=6, column=0, sticky=tk.W, pady=5, padx=5)
        self.side2_tag1 = tk.StringVar(value="bg")
        self.side2_tag1_entry = tk.Entry(temp_frame, textvariable=self.side2_tag1, width=15,
                                          font=("Consolas", 9), state='disabled')
        self.side2_tag1_entry.grid(row=6, column=1, sticky=tk.E, pady=5, padx=5)
        
        # HMR Side 2 Temp Tag 2
        side2_tag2_label = tk.Label(temp_frame, text="Side 2 Tag 2 (optional):", 
                                     font=("Consolas", 9, "bold"))
        side2_tag2_label.grid(row=7, column=0, sticky=tk.W, pady=5, padx=5)
        self.side2_tag2 = tk.StringVar(value="bj")
        self.side2_tag2_entry = tk.Entry(temp_frame, textvariable=self.side2_tag2, width=15,
                                          font=("Consolas", 9), state='disabled')
        self.side2_tag2_entry.grid(row=7, column=1, sticky=tk.E, pady=5, padx=5)
        
        # Flow rate inputs
        flow_frame = ttk.LabelFrame(left_panel, text="", padding="10")
        flow_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        flow_frame.grid_columnconfigure(0, weight=1)
        flow_frame.grid_columnconfigure(1, weight=1)
        
        flow_title = tk.Label(flow_frame, text="Volumetric Flow Rates (SLPM)", 
                             font=("Consolas", 10, "bold"))
        flow_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Preset dropdown
        preset_label = tk.Label(flow_frame, text="Preset:", 
                               font=("Consolas", 9, "bold"))
        preset_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        
        self.flow_preset = tk.StringVar(value="Manual")
        preset_menu = ttk.Combobox(flow_frame, textvariable=self.flow_preset,
                                  values=["Manual", "Bakeout", "RWGS", "Realtime"],
                                  state="readonly", width=12, font=("Consolas", 9))
        preset_menu.grid(row=1, column=1, sticky=tk.E, pady=5, padx=5)
        preset_menu.bind("<<ComboboxSelected>>", lambda e: self.apply_flow_preset())
        
        co2_label = tk.Label(flow_frame, text="CO₂ (SLPM):", 
                            font=("Consolas", 9, "bold"))
        co2_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.co2_flow = tk.StringVar(value="100")
        self.co2_entry = tk.Entry(flow_frame, textvariable=self.co2_flow, width=15,
                            font=("Consolas", 9))
        self.co2_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        h2_label = tk.Label(flow_frame, text="H₂ (SLPM):", 
                           font=("Consolas", 9, "bold"))
        h2_label.grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.h2_flow = tk.StringVar(value="100")
        self.h2_entry = tk.Entry(flow_frame, textvariable=self.h2_flow, width=15,
                           font=("Consolas", 9))
        self.h2_entry.grid(row=3, column=1, sticky=tk.E, pady=5, padx=5)
        
        n2_label = tk.Label(flow_frame, text="N₂ (SLPM):", 
                          font=("Consolas", 9, "bold"))
        n2_label.grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.n2_flow = tk.StringVar(value="0")
        self.n2_entry = tk.Entry(flow_frame, textvariable=self.n2_flow, width=15,
                           font=("Consolas", 9))
        self.n2_entry.grid(row=4, column=1, sticky=tk.E, pady=5, padx=5)
        
        # CO2 conversion
        conv_frame = ttk.LabelFrame(left_panel, text="", padding="10")
        conv_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        conv_frame.grid_columnconfigure(0, weight=1)
        conv_frame.grid_columnconfigure(1, weight=1)
        
        conv_title = tk.Label(conv_frame, text="CO₂ Conversion", 
                             font=("Consolas", 10, "bold"))
        conv_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        conv_label = tk.Label(conv_frame, text="CO₂ Conversion (%):", 
                             font=("Consolas", 9, "bold"))
        conv_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.co2_conversion = tk.StringVar(value="50")
        conv_entry = tk.Entry(conv_frame, textvariable=self.co2_conversion, width=15,
                            font=("Consolas", 9))
        conv_entry.grid(row=1, column=1, sticky=tk.E, pady=5, padx=5)
        
        # Ambient heat loss
        heatloss_frame = ttk.LabelFrame(left_panel, text="", padding="10")
        heatloss_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        heatloss_frame.grid_columnconfigure(0, weight=1)
        heatloss_frame.grid_columnconfigure(1, weight=1)
        
        heatloss_title = tk.Label(heatloss_frame, text="Ambient Heat Loss", 
                                 font=("Consolas", 10, "bold"))
        heatloss_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        heatloss_label = tk.Label(heatloss_frame, text="Heat Loss (W):", 
                                 font=("Consolas", 9, "bold"))
        heatloss_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.heat_loss = tk.StringVar(value="0")
        # Add trace to recalculate historical values when heat loss changes
        self.heat_loss.trace_add("write", lambda *args: self.recalculate_power_history())
        heatloss_entry = tk.Entry(heatloss_frame, textvariable=self.heat_loss, width=15,
                                  font=("Consolas", 9))
        heatloss_entry.grid(row=1, column=1, sticky=tk.E, pady=5, padx=5)
        
        # Calculate button and Live Update controls
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=4, column=0, pady=10)
        
        calc_button = ttk.Button(button_frame, text="Calculate Power", command=self.calculate)
        calc_button.grid(row=0, column=0, padx=5)
        
        self.live_update_button = ttk.Button(button_frame, text="Start Live Updates", 
                                             command=self.toggle_live_updates)
        self.live_update_button.grid(row=0, column=1, padx=5)
        
        if not self.db_available:
            self.live_update_button.config(state='disabled')
        
        # Results - centered at bottom
        result_frame = ttk.LabelFrame(left_panel, text="", padding="10")
        result_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_columnconfigure(1, weight=1)
        
        result_title = tk.Label(result_frame, text="Results", 
                               font=("Consolas", 10, "bold"))
        result_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        power_label = tk.Label(result_frame, text="Power Required:", 
                              font=("Consolas", 10, "bold"))
        power_label.grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        
        self.power_result = tk.StringVar(value="---")
        result_label = tk.Label(result_frame, textvariable=self.power_result, 
                               font=("Consolas", 12, "bold"),
                               foreground="blue")
        result_label.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        power_plus25_label = tk.Label(result_frame, text="+25% Power:", 
                                      font=("Consolas", 10, "bold"))
        power_plus25_label.grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)
        
        self.power_plus25_result = tk.StringVar(value="---")
        result_plus25_label = tk.Label(result_frame, textvariable=self.power_plus25_result, 
                                       font=("Consolas", 12, "bold"),
                                       foreground="blue")
        result_plus25_label.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        
        self.warning_result = tk.StringVar(value="")
        warning_label = tk.Label(result_frame, textvariable=self.warning_result, 
                                font=("Consolas", 9),
                                foreground="orange", wraplength=500)
        warning_label.grid(row=3, column=0, columnspan=2, pady=5, padx=5)
        warning_label.grid_remove()  # Hide initially since it's empty
        
        # Trace to show/hide warning label based on content
        def on_warning_change(*args):
            if self.warning_result.get().strip():
                warning_label.grid()
            else:
                warning_label.grid_remove()
        
        self.warning_result.trace_add("write", on_warning_change)
        
        # Plot controls
        plot_control_frame = ttk.LabelFrame(left_panel, text="", padding="10")
        plot_control_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        plot_control_frame.grid_columnconfigure(0, weight=1)
        plot_control_frame.grid_columnconfigure(1, weight=1)
        
        plot_control_title = tk.Label(plot_control_frame, text="Plot Time Range", 
                                      font=("Consolas", 10, "bold"))
        plot_control_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Radio buttons for time range selection
        self.time_range_mode = tk.StringVar(value="last")
        last_radio = tk.Radiobutton(plot_control_frame, text="Last", 
                                   variable=self.time_range_mode, value="last",
                                   font=("Consolas", 9), command=self.update_plot_range)
        last_radio.grid(row=1, column=0, sticky=tk.W, padx=5)
        
        self.time_value = tk.StringVar(value="60")
        time_value_entry = tk.Entry(plot_control_frame, textvariable=self.time_value, width=10,
                                   font=("Consolas", 9))
        time_value_entry.grid(row=1, column=1, sticky=tk.W, padx=5)
        time_value_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        self.time_unit = tk.StringVar(value="seconds")
        time_unit_menu = ttk.Combobox(plot_control_frame, textvariable=self.time_unit, 
                                      values=["seconds", "minutes", "hours"], width=10,
                                      state="readonly", font=("Consolas", 9))
        time_unit_menu.grid(row=1, column=2, sticky=tk.W, padx=5)
        time_unit_menu.bind("<<ComboboxSelected>>", lambda e: self.update_plot_range())
        
        custom_radio = tk.Radiobutton(plot_control_frame, text="Custom Range", 
                                     variable=self.time_range_mode, value="custom",
                                     font=("Consolas", 9), command=self.toggle_custom_range)
        custom_radio.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Custom range input fields (initially hidden)
        custom_range_frame = ttk.Frame(plot_control_frame)
        custom_range_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        start_label = tk.Label(custom_range_frame, text="Start:", font=("Consolas", 9, "bold"))
        start_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.custom_start = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        start_entry = tk.Entry(custom_range_frame, textvariable=self.custom_start, width=20,
                               font=("Consolas", 9))
        start_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        start_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        end_label = tk.Label(custom_range_frame, text="End:", font=("Consolas", 9, "bold"))
        end_label.grid(row=0, column=2, sticky=tk.W, padx=5)
        
        self.custom_end = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        end_entry = tk.Entry(custom_range_frame, textvariable=self.custom_end, width=20,
                             font=("Consolas", 9))
        end_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        end_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        # Store reference to custom_range_frame for show/hide
        self.custom_range_frame = custom_range_frame
        self.custom_range_frame.grid_remove()  # Initially hidden
        
        # Y-axis limits section
        yaxis_frame = ttk.LabelFrame(plot_control_frame, text="Y-Axis Limits (Optional)", padding="5")
        yaxis_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Power plot y-axis limits
        power_ylim_label = tk.Label(yaxis_frame, text="Power Plot:", font=("Consolas", 9, "bold"))
        power_ylim_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        power_ymin_label = tk.Label(yaxis_frame, text="Min (W):", font=("Consolas", 9))
        power_ymin_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.power_ymin = tk.StringVar(value="")
        power_ymin_entry = tk.Entry(yaxis_frame, textvariable=self.power_ymin, width=10,
                                    font=("Consolas", 9))
        power_ymin_entry.grid(row=0, column=2, sticky=tk.W, padx=5)
        power_ymin_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        power_ymax_label = tk.Label(yaxis_frame, text="Max (W):", font=("Consolas", 9))
        power_ymax_label.grid(row=0, column=3, sticky=tk.W, padx=5)
        self.power_ymax = tk.StringVar(value="")
        power_ymax_entry = tk.Entry(yaxis_frame, textvariable=self.power_ymax, width=10,
                                    font=("Consolas", 9))
        power_ymax_entry.grid(row=0, column=4, sticky=tk.W, padx=5)
        power_ymax_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        # Temperature plot y-axis limits
        temp_ylim_label = tk.Label(yaxis_frame, text="Temp Plot:", font=("Consolas", 9, "bold"))
        temp_ylim_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        temp_ymin_label = tk.Label(yaxis_frame, text="Min (°C):", font=("Consolas", 9))
        temp_ymin_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.temp_ymin = tk.StringVar(value="")
        temp_ymin_entry = tk.Entry(yaxis_frame, textvariable=self.temp_ymin, width=10,
                                   font=("Consolas", 9))
        temp_ymin_entry.grid(row=1, column=2, sticky=tk.W, padx=5)
        temp_ymin_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        temp_ymax_label = tk.Label(yaxis_frame, text="Max (°C):", font=("Consolas", 9))
        temp_ymax_label.grid(row=1, column=3, sticky=tk.W, padx=5)
        self.temp_ymax = tk.StringVar(value="")
        temp_ymax_entry = tk.Entry(yaxis_frame, textvariable=self.temp_ymax, width=10,
                                   font=("Consolas", 9))
        temp_ymax_entry.grid(row=1, column=4, sticky=tk.W, padx=5)
        temp_ymax_entry.bind('<KeyRelease>', lambda e: self.update_plot_range())
        
        # Power plot
        power_plot_frame = ttk.LabelFrame(right_panel, text="Power vs Time", padding="5")
        power_plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        power_plot_frame.grid_columnconfigure(0, weight=1)
        power_plot_frame.grid_rowconfigure(0, weight=1)
        
        self.power_fig = Figure(figsize=(10, 3), dpi=100)
        self.power_ax = self.power_fig.add_subplot(111)
        self.power_ax.set_xlabel("Time", fontsize=9)
        self.power_ax.set_ylabel("Power (W)", fontsize=9)
        self.power_ax.grid(True, alpha=0.3)
        self.power_canvas = FigureCanvasTkAgg(self.power_fig, power_plot_frame)
        self.power_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add hover annotation for power plot
        self.power_annotation = self.power_ax.annotate('', xy=(0, 0), xytext=(10, 10),
                                                       textcoords="offset points",
                                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                                       arrowprops=dict(arrowstyle='->'),
                                                       fontsize=8)
        self.power_annotation.set_visible(False)
        self.power_canvas.mpl_connect('motion_notify_event', lambda e: self.on_power_hover(e))
        
        # Temperature plot
        temp_plot_frame = ttk.LabelFrame(right_panel, text="Temperature vs Time", padding="5")
        temp_plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        temp_plot_frame.grid_columnconfigure(0, weight=1)
        temp_plot_frame.grid_rowconfigure(0, weight=1)
        
        self.temp_fig = Figure(figsize=(10, 3), dpi=100)
        self.temp_ax = self.temp_fig.add_subplot(111)
        self.temp_ax.set_xlabel("Time", fontsize=9)
        self.temp_ax.set_ylabel("Temperature (°C)", fontsize=9)
        self.temp_ax.grid(True, alpha=0.3)
        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, temp_plot_frame)
        self.temp_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add hover annotation for temperature plot
        self.temp_annotation = self.temp_ax.annotate('', xy=(0, 0), xytext=(10, 10),
                                                      textcoords="offset points",
                                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                                      arrowprops=dict(arrowstyle='->'),
                                                      fontsize=8)
        self.temp_annotation.set_visible(False)
        self.temp_canvas.mpl_connect('motion_notify_event', lambda e: self.on_temp_hover(e))
        
        # Store plot data for hover functionality
        self.power_plot_data = None
        self.temp_plot_data = None
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # Left panel - fixed width
        main_frame.columnconfigure(1, weight=1)  # Right panel - expands
        main_frame.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
    
    def toggle_db_mode(self):
        """Enable/disable database tag entry fields and set flow preset"""
        if self.use_db_temps.get():
            # Enable tag fields, disable manual entry
            self.side1_entry.config(state='disabled')
            self.side2_entry.config(state='disabled')
            self.side1_tag1_entry.config(state='normal')
            self.side1_tag2_entry.config(state='normal')
            self.side2_tag1_entry.config(state='normal')
            self.side2_tag2_entry.config(state='normal')
            # Automatically set flow preset to "Realtime"
            self.flow_preset.set("Realtime")
            self.apply_flow_preset()
        else:
            # Disable tag fields, enable manual entry
            self.side1_entry.config(state='normal')
            self.side2_entry.config(state='normal')
            self.side1_tag1_entry.config(state='disabled')
            self.side1_tag2_entry.config(state='disabled')
            self.side2_tag1_entry.config(state='disabled')
            self.side2_tag2_entry.config(state='disabled')
    
    def expand_tag_code(self, tag_input, side):
        """
        Convert two-letter tag code to full tag name
        
        Parameters:
        - tag_input: Two-letter code (e.g., "ag", "AG", "aG") or full tag name
        - side: 1 or 2 (for Side 1 or Side 2)
        
        Returns:
        - Full tag name (e.g., "ai/te_308ag/val") or original input if already full name
        """
        if not tag_input or not tag_input.strip():
            return None
        
        tag_input = tag_input.strip()
        
        # If it's already a full tag name (contains "/"), return as-is
        if "/" in tag_input:
            return tag_input
        
        # If it's exactly 2 characters, expand it
        if len(tag_input) == 2:
            # Convert to lowercase for consistency
            two_letter_code = tag_input.lower()
            # Construct full tag name: "ai/te_308" + two_letter_code + "/val"
            return f"ai/te_308{two_letter_code}/val"
        
        # If it's not 2 characters and not a full name, return as-is (might be invalid, but let user handle)
        return tag_input
    
    def get_tag_id_from_name(self, tag_name):
        """Convert tag name to tag ID"""
        if not self.db_available or not tag_name:
            return None
        # Reverse lookup: find ID for tag name
        for tag_id, name in self.tag_id_dict.items():
            if name == tag_name:
                return tag_id
        return None
    
    def fetch_temperatures_from_db(self):
        """Fetch current temperature values from database"""
        if not self.db_available or not self.use_db_temps.get():
            return None, None
        
        try:
            # Expand tag codes to full tag names
            side1_tag1_full = self.expand_tag_code(self.side1_tag1.get(), 1)
            side1_tag2_full = self.expand_tag_code(self.side1_tag2.get(), 1) if self.side1_tag2.get().strip() else None
            side2_tag1_full = self.expand_tag_code(self.side2_tag1.get(), 2)
            side2_tag2_full = self.expand_tag_code(self.side2_tag2.get(), 2) if self.side2_tag2.get().strip() else None
            
            # Get tag IDs
            side1_tag1_id = self.get_tag_id_from_name(side1_tag1_full) if side1_tag1_full else None
            side1_tag2_id = self.get_tag_id_from_name(side1_tag2_full) if side1_tag2_full else None
            side2_tag1_id = self.get_tag_id_from_name(side2_tag1_full) if side2_tag1_full else None
            side2_tag2_id = self.get_tag_id_from_name(side2_tag2_full) if side2_tag2_full else None
            
            # Get recent data (last 2 minutes to ensure we get data)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=2)
            
            # Fetch Side 1 temperatures
            side1_tag1_val = None
            side1_tag2_val = None
            if side1_tag1_id:
                tagids = [side1_tag1_id]
                if side1_tag2_id:
                    tagids.append(side1_tag2_id)
                
                df = self.db_handler.getDataframeBetween(start_time, end_time, tagids)
                if df is not None and not df.empty:
                    tag1_name = side1_tag1_full
                    if tag1_name in df.columns:
                        side1_tag1_val = df[tag1_name].dropna().iloc[-1] if not df[tag1_name].dropna().empty else None
                    
                    if side1_tag2_id:
                        tag2_name = side1_tag2_full
                        if tag2_name in df.columns:
                            side1_tag2_val = df[tag2_name].dropna().iloc[-1] if not df[tag2_name].dropna().empty else None
            
            # Fetch Side 2 temperatures
            side2_tag1_val = None
            side2_tag2_val = None
            if side2_tag1_id:
                tagids = [side2_tag1_id]
                if side2_tag2_id:
                    tagids.append(side2_tag2_id)
                
                df = self.db_handler.getDataframeBetween(start_time, end_time, tagids)
                if df is not None and not df.empty:
                    tag1_name = side2_tag1_full
                    if tag1_name in df.columns:
                        side2_tag1_val = df[tag1_name].dropna().iloc[-1] if not df[tag1_name].dropna().empty else None
                    
                    if side2_tag2_id:
                        tag2_name = side2_tag2_full
                        if tag2_name in df.columns:
                            side2_tag2_val = df[tag2_name].dropna().iloc[-1] if not df[tag2_name].dropna().empty else None
            
            # Calculate side temperatures based on the new logic:
            # Use max of whichever pair is higher on average, and min of whichever pair is lower on average
            side1_temp = None
            side2_temp = None
            
            # Get individual values (handle None cases)
            side1_vals = [v for v in [side1_tag1_val, side1_tag2_val] if v is not None]
            side2_vals = [v for v in [side2_tag1_val, side2_tag2_val] if v is not None]
            
            if side1_vals and side2_vals:
                # Calculate averages
                side1_avg = sum(side1_vals) / len(side1_vals) if side1_vals else None
                side2_avg = sum(side2_vals) / len(side2_vals) if side2_vals else None
                
                if side1_avg is not None and side2_avg is not None:
                    if side1_avg >= side2_avg:
                        # Side 1 is higher on average: use max for side1, min for side2
                        side1_temp = max(side1_vals)
                        side2_temp = min(side2_vals)
                    else:
                        # Side 2 is higher on average: use min for side1, max for side2
                        side1_temp = min(side1_vals)
                        side2_temp = max(side2_vals)
            elif side1_vals:
                # Only side1 has values - use the value(s) directly
                side1_temp = side1_vals[0] if len(side1_vals) == 1 else max(side1_vals)
            elif side2_vals:
                # Only side2 has values - use the value(s) directly
                side2_temp = side2_vals[0] if len(side2_vals) == 1 else max(side2_vals)
            
            return side1_temp, side2_temp
        except Exception as e:
            print(f"Error fetching temperatures from database: {e}")
            return None, None
    
    def fetch_realtime_flow_rates_from_db(self):
        """Fetch current flow rate values from database tags"""
        if not self.db_available:
            return None, None, None
        
        try:
            # Get tag IDs
            co2_tag_id = self.get_tag_id_from_name("ai/fi_116/val")
            h2_tag_id = self.get_tag_id_from_name("ai/fi_106m/val")
            n2_tag_id = self.get_tag_id_from_name("ai/fi_126m/val")
            
            # Get recent data (last 2 minutes to ensure we get data)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=2)
            
            co2_value = None
            h2_value = None
            n2_value = None
            
            # Fetch CO2 flow rate
            if co2_tag_id:
                df = self.db_handler.getDataframeBetween(start_time, end_time, [co2_tag_id])
                if df is not None and not df.empty:
                    tag_name = "ai/fi_116/val"
                    if tag_name in df.columns:
                        co2_value = df[tag_name].dropna().iloc[-1] if not df[tag_name].dropna().empty else None
            
            # Fetch H2 flow rate
            if h2_tag_id:
                df = self.db_handler.getDataframeBetween(start_time, end_time, [h2_tag_id])
                if df is not None and not df.empty:
                    tag_name = "ai/fi_106m/val"
                    if tag_name in df.columns:
                        h2_value = df[tag_name].dropna().iloc[-1] if not df[tag_name].dropna().empty else None
            
            # Fetch N2 flow rate
            if n2_tag_id:
                df = self.db_handler.getDataframeBetween(start_time, end_time, [n2_tag_id])
                if df is not None and not df.empty:
                    tag_name = "ai/fi_126m/val"
                    if tag_name in df.columns:
                        n2_value = df[tag_name].dropna().iloc[-1] if not df[tag_name].dropna().empty else None
            
            return co2_value, h2_value, n2_value
        except Exception as e:
            print(f"Error fetching realtime flow rates from database: {e}")
            return None, None, None
    
    def apply_flow_preset(self):
        """Apply selected flow rate preset"""
        preset = self.flow_preset.get()
        
        if preset == "Bakeout":
            self.co2_flow.set("0")
            self.h2_flow.set("0")
            self.n2_flow.set("150")
            # Enable fields for editing
            self.co2_entry.config(state='normal')
            self.h2_entry.config(state='normal')
            self.n2_entry.config(state='normal')
        elif preset == "RWGS":
            self.co2_flow.set("174")
            self.h2_flow.set("400")
            self.n2_flow.set("0")
            # Enable fields for editing
            self.co2_entry.config(state='normal')
            self.h2_entry.config(state='normal')
            self.n2_entry.config(state='normal')
        elif preset == "Realtime":
            # Fetch from database
            co2_val, h2_val, n2_val = self.fetch_realtime_flow_rates_from_db()
            # Display values immediately (use 0 if None)
            if co2_val is not None:
                self.co2_flow.set(f"{co2_val:.2f}")
            else:
                self.co2_flow.set("0")
            if h2_val is not None:
                self.h2_flow.set(f"{h2_val:.2f}")
            else:
                self.h2_flow.set("0")
            if n2_val is not None:
                self.n2_flow.set(f"{n2_val:.2f}")
            else:
                self.n2_flow.set("0")
            # Disable fields - values are read-only from database
            self.co2_entry.config(state='readonly')
            self.h2_entry.config(state='readonly')
            self.n2_entry.config(state='readonly')
        else:  # "Manual"
            # Enable fields for manual editing
            self.co2_entry.config(state='normal')
            self.h2_entry.config(state='normal')
            self.n2_entry.config(state='normal')
    
    def fetch_realtime_power_from_db(self):
        """Fetch current realtime power value from database tag ai/ji_308/val"""
        if not self.db_available:
            return None
        
        try:
            # Get tag ID for ai/ji_308/val
            tag_id = self.get_tag_id_from_name("ai/ji_308/val")
            if tag_id is None:
                return None
            
            # Get recent data (last 2 minutes to ensure we get data)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=2)
            
            # Fetch realtime power
            df = self.db_handler.getDataframeBetween(start_time, end_time, [tag_id])
            if df is not None and not df.empty:
                tag_name = "ai/ji_308/val"
                if tag_name in df.columns:
                    realtime_power = df[tag_name].dropna().iloc[-1] if not df[tag_name].dropna().empty else None
                    return realtime_power
            
            return None
        except Exception as e:
            print(f"Error fetching realtime power from database: {e}")
            return None
    
    def toggle_live_updates(self):
        """Start or stop live updates"""
        if self.live_update_active:
            # Stop live updates
            self.live_update_active = False
            self.live_update_button.config(text="Start Live Updates")
            if self.update_thread and self.update_thread.is_alive():
                # Thread will stop on next iteration
                pass
        else:
            # Start live updates
            if not self.use_db_temps.get():
                self.warning_result.set("Enable 'Realtime' to start live updates")
                return
            
            if not self.side1_tag1.get().strip() or not self.side2_tag1.get().strip():
                self.warning_result.set("Enter at least one tag for each temperature")
                return
            
            self.live_update_active = True
            self.live_update_button.config(text="Stop Live Updates")
            self.update_thread = threading.Thread(target=self.live_update_loop, daemon=True)
            self.update_thread.start()
    
    def live_update_loop(self):
        """Background thread that updates calculations 4 times per second"""
        while self.live_update_active:
            try:
                # Fetch temperatures from database
                side1_temp, side2_temp = self.fetch_temperatures_from_db()
                
                # Fetch realtime power from database
                realtime_power = self.fetch_realtime_power_from_db()
                
                # Update flow rates if "Realtime" preset is selected
                if self.flow_preset.get() == "Realtime":
                    co2_val, h2_val, n2_val = self.fetch_realtime_flow_rates_from_db()
                    if co2_val is not None:
                        self.root.after(0, lambda c=co2_val: self.co2_flow.set(f"{c:.2f}"))
                    if h2_val is not None:
                        self.root.after(0, lambda h=h2_val: self.h2_flow.set(f"{h:.2f}"))
                    if n2_val is not None:
                        self.root.after(0, lambda n=n2_val: self.n2_flow.set(f"{n:.2f}"))
                
                if side1_temp is not None and side2_temp is not None:
                    # Update temperature display fields (read-only)
                    # Use default args to avoid closure issues
                    self.root.after(0, lambda s1=side1_temp: self.side1_temp.set(f"{s1:.2f}"))
                    self.root.after(0, lambda s2=side2_temp: self.side2_temp.set(f"{s2:.2f}"))
                    
                    # Calculate power with current settings
                    self.root.after(0, lambda s1=side1_temp, s2=side2_temp, rp=realtime_power: self.calculate_with_temps(s1, s2, rp))
                
                # Update flow plot and temp/pressure if we're on the conversion tab
                try:
                    current_tab = self.notebook.index(self.notebook.select())
                    if current_tab == 1:  # Flow conversion tab is index 1
                        self.root.after(0, self.update_flow_plot)
                        # Also update temperature and pressure values in conversion tab
                        if self.use_db_conv.get():
                            self.root.after(0, self.update_conv_temp_pressure)
                except:
                    pass
                
                time.sleep(0.25)  # Update 4 times per second (every 0.25 seconds)
            except Exception as e:
                print(f"Error in live update loop: {e}")
                time.sleep(0.25)
    
    def calculate_with_temps(self, side1_temp_val, side2_temp_val, realtime_power=None):
        """Calculate power with given temperature values (used by live updates)"""
        try:
            # Determine inlet and outlet based on which is higher
            # Higher temp = outlet, lower temp = inlet (gas always heats up)
            inlet_temp = min(side1_temp_val, side2_temp_val)
            outlet_temp = max(side1_temp_val, side2_temp_val)
            
            # Get other inputs
            co2_slpm = float(self.co2_flow.get())
            h2_slpm = float(self.h2_flow.get())
            n2_slpm = float(self.n2_flow.get())
            co2_conversion = float(self.co2_conversion.get())
            heat_loss = float(self.heat_loss.get())
            
            # Calculate power
            power = self.calculator.calculate_power(
                inlet_temp, outlet_temp,
                co2_slpm, h2_slpm, n2_slpm, co2_conversion
            )
            
            # Add ambient heat loss
            total_power = power + heat_loss
            
            # Calculate +25% power
            power_plus_25 = total_power * 1.25
            
            # Store data point (store base power without heat loss so we can recalculate)
            now = datetime.now()
            with self.update_lock:
                # Store: (datetime, base_power_without_heat_loss, total_power, power_plus_25)
                self.power_history.append((now, power, total_power, power_plus_25))
                self.temp_history.append((now, side1_temp_val, side2_temp_val))
                
                # Store realtime power if available (convert from kW to W)
                if realtime_power is not None:
                    realtime_power_w = realtime_power * 1000  # Convert kW to W
                    self.realtime_power_history.append((now, realtime_power_w))
            
            # Update display
            if total_power >= 1000:
                power_kw = total_power / 1000
                self.power_result.set(f"{power_kw:.2f} kW ({total_power:.1f} W)")
            else:
                self.power_result.set(f"{total_power:.1f} W")
            
            if power_plus_25 >= 1000:
                power_plus25_kw = power_plus_25 / 1000
                self.power_plus25_result.set(f"{power_plus25_kw:.2f} kW ({power_plus_25:.1f} W)")
            else:
                self.power_plus25_result.set(f"{power_plus_25:.1f} W")
            
            # Update plots
            self.root.after(0, self.update_plots)
            
        except Exception as e:
            print(f"Error in calculate_with_temps: {e}")
            self.root.after(0, lambda: self.power_result.set("Error"))
            self.root.after(0, lambda: self.power_plus25_result.set("---"))
    
    def toggle_custom_range(self):
        """Show/hide custom range input fields"""
        if self.time_range_mode.get() == "custom":
            self.custom_range_frame.grid()
        else:
            self.custom_range_frame.grid_remove()
        self.update_plots()
    
    def update_plot_range(self):
        """Update plot x-axis range based on user selection"""
        self.update_plots()
    
    def recalculate_power_history(self):
        """Recalculate all historical power values when heat loss changes"""
        try:
            heat_loss = float(self.heat_loss.get())
            with self.update_lock:
                # Recalculate all power values with new heat loss
                recalculated = []
                for dt, base_power, _, _ in self.power_history:
                    total_power = base_power + heat_loss
                    power_plus_25 = total_power * 1.25
                    recalculated.append((dt, base_power, total_power, power_plus_25))
                # Replace history
                self.power_history.clear()
                self.power_history.extend(recalculated)
            # Update plots and display
            self.root.after(0, self.update_plots)
            # Update current display if we have recent data
            if self.power_history:
                _, _, total_power, power_plus_25 = self.power_history[-1]
                if total_power >= 1000:
                    power_kw = total_power / 1000
                    self.power_result.set(f"{power_kw:.2f} kW ({total_power:.1f} W)")
                else:
                    self.power_result.set(f"{total_power:.1f} W")
                
                if power_plus_25 >= 1000:
                    power_plus25_kw = power_plus_25 / 1000
                    self.power_plus25_result.set(f"{power_plus25_kw:.2f} kW ({power_plus_25:.1f} W)")
                else:
                    self.power_plus25_result.set(f"{power_plus_25:.1f} W")
        except ValueError:
            # Invalid heat loss value, ignore
            pass
        except Exception as e:
            print(f"Error recalculating power history: {e}")
    
    def update_plots(self):
        """Update both power and temperature plots"""
        with self.update_lock:
            # Get data (create copies to avoid holding lock during plotting)
            power_times = [t for t, _, _, _ in self.power_history]
            power_values = [p for _, _, p, _ in self.power_history]
            power_plus25_values = [p25 for _, _, _, p25 in self.power_history]
            temp_times = [t for t, _, _ in self.temp_history]
            side1_temps = [t1 for _, t1, _ in self.temp_history]
            side2_temps = [t2 for _, _, t2 in self.temp_history]
            realtime_power_times = [t for t, _ in self.realtime_power_history]
            realtime_power_values = [p for _, p in self.realtime_power_history]
        
        # If no data, clear plots and return
        if not power_times and not temp_times and not realtime_power_times:
            self.power_ax.clear()
            self.power_ax.set_xlabel("Time", fontsize=9)
            self.power_ax.set_ylabel("Power (W)", fontsize=9)
            self.power_ax.grid(True, alpha=0.3)
            self.power_fig.tight_layout()
            self.power_canvas.draw()
            
            self.temp_ax.clear()
            self.temp_ax.set_xlabel("Time", fontsize=9)
            self.temp_ax.set_ylabel("Temperature (°C)", fontsize=9)
            self.temp_ax.grid(True, alpha=0.3)
            self.temp_fig.tight_layout()
            self.temp_canvas.draw()
            return
        
        # Determine x-axis range
        if self.time_range_mode.get() == "last":
            try:
                time_val = float(self.time_value.get())
                unit = self.time_unit.get()
                
                if unit == "seconds":
                    delta = timedelta(seconds=time_val)
                elif unit == "minutes":
                    delta = timedelta(minutes=time_val)
                else:  # hours
                    delta = timedelta(hours=time_val)
                
                # Use most recent time from any dataset
                all_times = power_times + temp_times + realtime_power_times
                if all_times:
                    x_max = max(all_times)
                    x_min = x_max - delta
                else:
                    x_min = datetime.now() - delta
                    x_max = datetime.now()
            except:
                # Default to last hour if error
                all_times = power_times + temp_times + realtime_power_times
                if all_times:
                    x_max = max(all_times)
                    x_min = x_max - timedelta(hours=1)
                else:
                    x_max = datetime.now()
                    x_min = x_max - timedelta(hours=1)
        else:  # custom range
            try:
                # Parse custom datetime strings
                start_str = self.custom_start.get().strip()
                end_str = self.custom_end.get().strip()
                
                # Try multiple datetime formats
                formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", 
                          "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"]
                
                x_min = None
                x_max = None
                
                for fmt in formats:
                    try:
                        if start_str:
                            x_min = datetime.strptime(start_str, fmt)
                        if end_str:
                            x_max = datetime.strptime(end_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if x_min is None or x_max is None:
                    raise ValueError("Could not parse datetime")
                
                if x_min >= x_max:
                    raise ValueError("Start time must be before end time")
                    
            except Exception as e:
                # Default to last hour if error
                print(f"Error parsing custom range: {e}")
                all_times = power_times + temp_times + realtime_power_times
                if all_times:
                    x_max = max(all_times)
                    x_min = x_max - timedelta(hours=1)
                else:
                    x_max = datetime.now()
                    x_min = x_max - timedelta(hours=1)
        
        # Filter data to x-axis range
        power_filtered = [(t, p, p25) for t, p, p25 in zip(power_times, power_values, power_plus25_values) if x_min <= t <= x_max]
        temp_filtered = [(t, t1, t2) for t, t1, t2 in zip(temp_times, side1_temps, side2_temps) if x_min <= t <= x_max]
        realtime_power_filtered = [(t, p) for t, p in zip(realtime_power_times, realtime_power_values) if x_min <= t <= x_max]
        
        # Update power plot
        self.power_ax.clear()
        # Recreate annotation after clearing
        self.power_annotation = self.power_ax.annotate('', xy=(0, 0), xytext=(10, 10),
                                                       textcoords="offset points",
                                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                                       arrowprops=dict(arrowstyle='->'),
                                                       fontsize=8)
        self.power_annotation.set_visible(False)
        
        all_powers = []
        self.power_plot_data = []  # Store for hover functionality
        
        if power_filtered:
            times, powers, powers_plus25 = zip(*power_filtered)
            line1 = self.power_ax.plot(times, powers, 'b-', label='Power Required', linewidth=1.5)[0]
            line2 = self.power_ax.plot(times, powers_plus25, 'g-', label='+25% Power', linewidth=1.5)[0]
            all_powers.extend(powers)
            all_powers.extend(powers_plus25)
            # Store data for hover
            self.power_plot_data.append(('Power Required', list(times), list(powers)))
            self.power_plot_data.append(('+25% Power', list(times), list(powers_plus25)))
        
        if realtime_power_filtered:
            times_rt, powers_rt = zip(*realtime_power_filtered)
            line3 = self.power_ax.plot(times_rt, powers_rt, 'r-', label='Realtime Power', linewidth=1.5)[0]
            all_powers.extend(powers_rt)
            # Store data for hover
            self.power_plot_data.append(('Realtime Power', list(times_rt), list(powers_rt)))
        
        if power_filtered or realtime_power_filtered:
            self.power_ax.legend(fontsize=8)
        
        self.power_ax.set_xlabel("Time", fontsize=9)
        self.power_ax.set_ylabel("Power (W)", fontsize=9)
        self.power_ax.grid(True, alpha=0.3)
        self.power_ax.set_xlim(x_min, x_max)
        # Set y-axis limits (use user input if provided, otherwise auto-scale)
        if all_powers:
            try:
                ymin_str = self.power_ymin.get().strip()
                ymax_str = self.power_ymax.get().strip()
                if ymin_str and ymax_str:
                    ymin = float(ymin_str)
                    ymax = float(ymax_str)
                    if ymin < ymax:
                        self.power_ax.set_ylim(ymin, ymax)
                    else:
                        self.power_ax.set_ylim(min(all_powers) * 0.95, max(all_powers) * 1.05)
                else:
                    self.power_ax.set_ylim(min(all_powers) * 0.95, max(all_powers) * 1.05)
            except ValueError:
                # Invalid input, use auto-scale
                self.power_ax.set_ylim(min(all_powers) * 0.95, max(all_powers) * 1.05)
        self.power_fig.tight_layout()
        self.power_canvas.draw()
        
        # Update temperature plot
        self.temp_ax.clear()
        # Recreate annotation after clearing
        self.temp_annotation = self.temp_ax.annotate('', xy=(0, 0), xytext=(10, 10),
                                                      textcoords="offset points",
                                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                                                      arrowprops=dict(arrowstyle='->'),
                                                      fontsize=8)
        self.temp_annotation.set_visible(False)
        
        self.temp_plot_data = []  # Store for hover functionality
        
        if temp_filtered:
            times, side1s, side2s = zip(*temp_filtered)
            self.temp_ax.plot(times, side1s, 'g-', label='HMR Side 1', linewidth=1.5)
            self.temp_ax.plot(times, side2s, 'r-', label='HMR Side 2', linewidth=1.5)
            self.temp_ax.legend(fontsize=8)
            # Store data for hover
            self.temp_plot_data.append(('HMR Side 1', list(times), list(side1s)))
            self.temp_plot_data.append(('HMR Side 2', list(times), list(side2s)))
        self.temp_ax.set_xlabel("Time", fontsize=9)
        self.temp_ax.set_ylabel("Temperature (°C)", fontsize=9)
        self.temp_ax.grid(True, alpha=0.3)
        self.temp_ax.set_xlim(x_min, x_max)
        # Set y-axis limits (use user input if provided, otherwise auto-scale)
        if temp_filtered:
            all_temps = list(side1s) + list(side2s)
            try:
                ymin_str = self.temp_ymin.get().strip()
                ymax_str = self.temp_ymax.get().strip()
                if ymin_str and ymax_str:
                    ymin = float(ymin_str)
                    ymax = float(ymax_str)
                    if ymin < ymax:
                        self.temp_ax.set_ylim(ymin, ymax)
                    else:
                        self.temp_ax.set_ylim(min(all_temps) * 0.95, max(all_temps) * 1.05)
                else:
                    self.temp_ax.set_ylim(min(all_temps) * 0.95, max(all_temps) * 1.05)
            except ValueError:
                # Invalid input, use auto-scale
                self.temp_ax.set_ylim(min(all_temps) * 0.95, max(all_temps) * 1.05)
        self.temp_fig.tight_layout()
        self.temp_canvas.draw()
    
    def on_power_hover(self, event):
        """Handle mouse hover on power plot"""
        if event.inaxes != self.power_ax or not self.power_plot_data:
            self.power_annotation.set_visible(False)
            self.power_canvas.draw_idle()
            return
        
        # Convert mouse position to data coordinates
        x_data = event.xdata
        y_data = event.ydata
        
        if x_data is None or y_data is None:
            self.power_annotation.set_visible(False)
            self.power_canvas.draw_idle()
            return
        
        # Find closest data point
        min_dist = float('inf')
        closest_label = None
        closest_y = None
        closest_x = None
        
        for label, times, values in self.power_plot_data:
            if not times or not values:
                continue
            # Find closest time point
            times_array = [t.timestamp() if isinstance(t, datetime) else t for t in times]
            x_data_ts = x_data.timestamp() if isinstance(x_data, datetime) else x_data
            
            # Find index of closest time
            times_np = np.array(times_array)
            idx = np.abs(times_np - x_data_ts).argmin()
            
            # Calculate distance
            dist = abs(times_np[idx] - x_data_ts)
            if dist < min_dist:
                min_dist = dist
                closest_label = label
                closest_y = values[idx]
                closest_x = times[idx]
        
        if closest_label is not None and min_dist < (self.power_ax.get_xlim()[1] - self.power_ax.get_xlim()[0]) / 50:
            # Show annotation
            self.power_annotation.xy = (closest_x, closest_y)
            self.power_annotation.set_text(f'{closest_label}\n{closest_y:.1f} W')
            self.power_annotation.set_visible(True)
        else:
            self.power_annotation.set_visible(False)
        
        self.power_canvas.draw_idle()
    
    def on_temp_hover(self, event):
        """Handle mouse hover on temperature plot"""
        if event.inaxes != self.temp_ax or not self.temp_plot_data:
            self.temp_annotation.set_visible(False)
            self.temp_canvas.draw_idle()
            return
        
        # Convert mouse position to data coordinates
        x_data = event.xdata
        y_data = event.ydata
        
        if x_data is None or y_data is None:
            self.temp_annotation.set_visible(False)
            self.temp_canvas.draw_idle()
            return
        
        # Find closest data point
        min_dist = float('inf')
        closest_label = None
        closest_y = None
        closest_x = None
        
        for label, times, values in self.temp_plot_data:
            if not times or not values:
                continue
            # Find closest time point
            times_array = [t.timestamp() if isinstance(t, datetime) else t for t in times]
            x_data_ts = x_data.timestamp() if isinstance(x_data, datetime) else x_data
            
            # Find index of closest time
            times_np = np.array(times_array)
            idx = np.abs(times_np - x_data_ts).argmin()
            
            # Calculate distance
            dist = abs(times_np[idx] - x_data_ts)
            if dist < min_dist:
                min_dist = dist
                closest_label = label
                closest_y = values[idx]
                closest_x = times[idx]
        
        if closest_label is not None and min_dist < (self.temp_ax.get_xlim()[1] - self.temp_ax.get_xlim()[0]) / 50:
            # Show annotation
            self.temp_annotation.xy = (closest_x, closest_y)
            self.temp_annotation.set_text(f'{closest_label}\n{closest_y:.1f} °C')
            self.temp_annotation.set_visible(True)
        else:
            self.temp_annotation.set_visible(False)
        
        self.temp_canvas.draw_idle()
    
    def create_flow_conversion_tab(self, parent):
        """Create widgets for the flow rate conversion tab"""
        # Title
        title_label = tk.Label(parent, text="Flow Rate Conversion", 
                              font=("Consolas", 16, "bold"))
        title_label.grid(row=0, column=0, pady=15)
        
        # Main content frame
        content_frame = ttk.Frame(parent, padding="10")
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
        # Gas selection buttons
        gas_frame = ttk.LabelFrame(content_frame, text="Select Gas", padding="10")
        gas_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.selected_gas = tk.StringVar(value="CO2")
        co2_radio = tk.Radiobutton(gas_frame, text="CO₂", variable=self.selected_gas, value="CO2",
                                   font=("Consolas", 9), command=self.switch_gas)
        co2_radio.grid(row=0, column=0, padx=10, pady=5)
        
        n2_radio = tk.Radiobutton(gas_frame, text="N₂", variable=self.selected_gas, value="N2",
                                  font=("Consolas", 9), command=self.switch_gas)
        n2_radio.grid(row=0, column=1, padx=10, pady=5)
        
        h2_radio = tk.Radiobutton(gas_frame, text="H₂", variable=self.selected_gas, value="H2",
                                  font=("Consolas", 9), command=self.switch_gas)
        h2_radio.grid(row=0, column=2, padx=10, pady=5)
        
        # Input section
        input_frame = ttk.LabelFrame(content_frame, text="Input", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        slpm_label = tk.Label(input_frame, text="SLPM (at STP: 0°C, 0 bar gauge):", 
                              font=("Consolas", 10, "bold"))
        slpm_label.grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        
        self.slpm_input = tk.StringVar(value="100")
        slpm_entry = tk.Entry(input_frame, textvariable=self.slpm_input, width=15,
                             font=("Consolas", 9))
        slpm_entry.grid(row=0, column=1, sticky=tk.E, pady=5, padx=5)
        
        calc_button = ttk.Button(input_frame, text="Calculate", command=self.calculate_flow_conversion)
        calc_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Output section
        output_frame = ttk.LabelFrame(content_frame, text="Output", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        lpm_label = tk.Label(output_frame, text="LPM at Actual Conditions:", 
                            font=("Consolas", 10, "bold"))
        lpm_label.grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        
        self.lpm_output = tk.StringVar(value="---")
        lpm_display = tk.Label(output_frame, textvariable=self.lpm_output, 
                              font=("Consolas", 12, "bold"), foreground="blue")
        lpm_display.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Temperature and Pressure section
        cond_frame = ttk.LabelFrame(content_frame, text="Conditions", padding="10")
        cond_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        cond_frame.grid_columnconfigure(0, weight=1)
        cond_frame.grid_columnconfigure(1, weight=1)
        
        # Realtime checkbox (controls both temperature and pressure)
        self.use_db_conv = tk.BooleanVar(value=True)
        realtime_check = tk.Checkbutton(cond_frame, text="Realtime",
                                        variable=self.use_db_conv,
                                        font=("Consolas", 9),
                                        command=self.toggle_conv_db_mode)
        realtime_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        # Temperature
        temp_title = tk.Label(cond_frame, text="Temperature", font=("Consolas", 10, "bold"))
        temp_title.grid(row=1, column=0, columnspan=2, pady=(10, 10))
        
        temp_manual_label = tk.Label(cond_frame, text="Temperature (°C):", 
                                     font=("Consolas", 9, "bold"))
        temp_manual_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.temp_manual = tk.StringVar(value="20")
        self.temp_manual_entry = tk.Entry(cond_frame, textvariable=self.temp_manual, width=15,
                                          font=("Consolas", 9), state='disabled')
        self.temp_manual_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        self.temp_db_tag_label = tk.Label(cond_frame, text="Tag: ai/ti_116/val", 
                                    font=("Consolas", 8), foreground="gray")
        self.temp_db_tag_label.grid(row=3, column=0, sticky=tk.W, pady=2, padx=5)
        
        self.temp_db_value = tk.StringVar(value="---")
        temp_db_display = tk.Label(cond_frame, textvariable=self.temp_db_value, 
                                   font=("Consolas", 9), foreground="blue")
        temp_db_display.grid(row=3, column=1, sticky=tk.W, pady=2, padx=5)
        
        self.temp_age_label = tk.Label(cond_frame, text="", 
                                        font=("Consolas", 7), foreground="gray")
        self.temp_age_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=0, padx=5)
        
        # Pressure
        pressure_title = tk.Label(cond_frame, text="Pressure", font=("Consolas", 10, "bold"))
        pressure_title.grid(row=5, column=0, columnspan=2, pady=(10, 10))
        
        pressure_manual_label = tk.Label(cond_frame, text="Pressure (bar gauge):", 
                                         font=("Consolas", 9, "bold"))
        pressure_manual_label.grid(row=6, column=0, sticky=tk.W, pady=5, padx=5)
        self.pressure_manual = tk.StringVar(value="0")
        self.pressure_manual_entry = tk.Entry(cond_frame, textvariable=self.pressure_manual, width=15,
                                             font=("Consolas", 9), state='disabled')
        self.pressure_manual_entry.grid(row=6, column=1, sticky=tk.E, pady=5, padx=5)
        
        self.pressure_db_tag_label = tk.Label(cond_frame, text="Tag: ai/pi_116/val (bar gauge)", 
                                        font=("Consolas", 8), foreground="gray")
        self.pressure_db_tag_label.grid(row=7, column=0, sticky=tk.W, pady=2, padx=5)
        
        self.pressure_db_value = tk.StringVar(value="---")
        pressure_db_display = tk.Label(cond_frame, textvariable=self.pressure_db_value, 
                                       font=("Consolas", 9), foreground="blue")
        pressure_db_display.grid(row=7, column=1, sticky=tk.W, pady=2, padx=5)
        
        self.pressure_age_label = tk.Label(cond_frame, text="", 
                                            font=("Consolas", 7), foreground="gray")
        self.pressure_age_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=0, padx=5)
        
        # Flow Rate Plot Controls
        plot_control_frame = ttk.LabelFrame(content_frame, text="Plot Controls", padding="5")
        plot_control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        # Radio buttons for time range selection
        self.flow_time_range_mode = tk.StringVar(value="last")
        flow_last_radio = tk.Radiobutton(plot_control_frame, text="Last", 
                                         variable=self.flow_time_range_mode, value="last",
                                         font=("Consolas", 9), command=self.update_flow_plot)
        flow_last_radio.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.flow_time_value = tk.StringVar(value="5")
        flow_time_value_entry = tk.Entry(plot_control_frame, textvariable=self.flow_time_value, width=10,
                                         font=("Consolas", 9))
        flow_time_value_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        flow_time_value_entry.bind('<KeyRelease>', lambda e: self.update_flow_plot())
        
        self.flow_time_unit = tk.StringVar(value="minutes")
        flow_time_unit_menu = ttk.Combobox(plot_control_frame, textvariable=self.flow_time_unit, 
                                          values=["seconds", "minutes", "hours"], width=10,
                                          state="readonly", font=("Consolas", 9))
        flow_time_unit_menu.grid(row=0, column=2, sticky=tk.W, padx=5)
        flow_time_unit_menu.bind("<<ComboboxSelected>>", lambda e: self.update_flow_plot())
        
        # Flow Rate Plot
        self.plot_frame = ttk.LabelFrame(content_frame, text="Flow Rate vs Time", padding="5")
        self.plot_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        
        self.flow_fig = Figure(figsize=(10, 4), dpi=100)
        self.flow_ax = self.flow_fig.add_subplot(111)
        self.flow_ax.set_xlabel("Time", fontsize=9)
        self.flow_ax.set_ylabel("Flow Rate (LPM)", fontsize=9)
        self.flow_ax.grid(True, alpha=0.3)
        self.flow_canvas = FigureCanvasTkAgg(self.flow_fig, self.plot_frame)
        self.flow_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(4, weight=1)
    
    def get_gas_tags(self, gas):
        """Get temperature, pressure, and flow rate tags for the selected gas
        Returns: (temp_tag_name, pressure_tag_name, flow_tag_id, gas_name)
        Note: flow_tag_id is the actual tag ID number, not the tag name"""
        if gas == "CO2":
            return ("ai/ti_116/val", "ai/pi_116/val", None, "CO₂")  # flow_tag_id will be set by get_gas_flow_tag_id
        elif gas == "N2":
            return ("ai/ti_126/val", "ai/pi_126/val", None, "N₂")
        elif gas == "H2":
            return ("ai/ti_106/val", "ai/pi_106/val", None, "H₂")
        else:
            return ("ai/ti_116/val", "ai/pi_116/val", None, "CO₂")
    
    def get_gas_flow_tag_id(self, gas):
        """Get the flow rate tag ID for the selected gas"""
        if gas == "CO2":
            return 299  # Tag ID 299 for CO2
        elif gas == "N2":
            return None  # N2 flow rate not available
        elif gas == "H2":
            return 298  # Tag ID 298 for hydrogen
        else:
            return None
    
    def switch_gas(self):
        """Update UI and plot when gas selection changes"""
        gas = self.selected_gas.get()
        temp_tag, pressure_tag, _, gas_name = self.get_gas_tags(gas)
        flow_tag_id = self.get_gas_flow_tag_id(gas)
        
        # Update tag labels
        self.temp_db_tag_label.config(text=f"Tag: {temp_tag}")
        self.pressure_db_tag_label.config(text=f"Tag: {pressure_tag} (bar gauge)")
        if flow_tag_id is not None:
            self.plot_frame.config(text=f"{gas_name} Flow Rate vs Time (Tag ID: {flow_tag_id})")
        else:
            self.plot_frame.config(text=f"{gas_name} Flow Rate vs Time")
        
        # If Realtime is enabled, update conditions and output
        if self.use_db_conv.get():
            self.toggle_conv_db_mode()  # This will fetch and update the values
            # Also update the output if we have SLPM input
            try:
                slpm = float(self.slpm_input.get())
                if slpm > 0:
                    self.calculate_flow_conversion()
            except:
                pass
        
        # Update plot
        self.update_flow_plot()
    
    def update_conv_temp_pressure(self):
        """Update temperature and pressure values from database (called by live update loop)"""
        if self.use_db_conv.get() and self.db_available:
            temp_c, temp_age, pressure_bar_gauge, pressure_age = self.fetch_temp_pressure_from_db()
            if temp_c is not None:
                if temp_age is not None and temp_age.total_seconds() > 60:  # Show age if > 1 minute
                    age_str = self.format_age(temp_age)
                    self.temp_db_value.set(f"{temp_c:.2f} °C")
                    self.temp_age_label.config(text=f"(Age: {age_str})")
                else:
                    self.temp_db_value.set(f"{temp_c:.2f} °C")
                    self.temp_age_label.config(text="")
            else:
                self.temp_db_value.set("---")
                self.temp_age_label.config(text="")
            if pressure_bar_gauge is not None:
                if pressure_age is not None and pressure_age.total_seconds() > 60:  # Show age if > 1 minute
                    age_str = self.format_age(pressure_age)
                    self.pressure_db_value.set(f"{pressure_bar_gauge:.2f} bar gauge")
                    self.pressure_age_label.config(text=f"(Age: {age_str})")
                else:
                    self.pressure_db_value.set(f"{pressure_bar_gauge:.2f} bar gauge")
                    self.pressure_age_label.config(text="")
            else:
                self.pressure_db_value.set("---")
                self.pressure_age_label.config(text="")
    
    def toggle_conv_db_mode(self):
        """Enable/disable manual entry fields for temperature and pressure in conversion tab"""
        if self.use_db_conv.get():
            self.temp_manual_entry.config(state='disabled')
            self.pressure_manual_entry.config(state='disabled')
            # Update display with database value
            self.update_conv_temp_pressure()
        else:
            self.temp_manual_entry.config(state='normal')
            self.pressure_manual_entry.config(state='normal')
            self.temp_db_value.set("---")
            self.pressure_db_value.set("---")
            self.temp_age_label.config(text="")
            self.pressure_age_label.config(text="")
    
    def format_age(self, age_delta):
        """Format a timedelta as a human-readable age string"""
        total_seconds = int(age_delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def fetch_temp_pressure_from_db(self):
        """Fetch temperature and pressure from database for flow conversion
        Returns: (temp_value, temp_age, pressure_value, pressure_age)
        If no value at current time, uses most recent value and returns its age"""
        if not self.db_available:
            return None, None, None, None
        
        try:
            # Get tags for selected gas
            gas = self.selected_gas.get()
            temp_tag, pressure_tag, _, _ = self.get_gas_tags(gas)
            
            # Get tag IDs
            temp_tag_id = self.get_tag_id_from_name(temp_tag)
            pressure_tag_id = self.get_tag_id_from_name(pressure_tag)
            
            # Get recent data (last hour to find most recent value)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            temp_value = None
            temp_age = None
            pressure_value = None
            pressure_age = None
            
            # Fetch temperature
            if temp_tag_id:
                df = self.db_handler.getDataframeBetween(start_time, end_time, [temp_tag_id])
                if df is not None and not df.empty:
                    if temp_tag in df.columns:
                        # Get non-null values
                        temp_series = df[temp_tag].dropna()
                        if not temp_series.empty:
                            temp_value = temp_series.iloc[-1]
                            # Get timestamp of this value
                            if 'datetime' in df.columns:
                                temp_timestamp = df.loc[temp_series.index[-1], 'datetime']
                                try:
                                    from pandas import Timestamp
                                    if isinstance(temp_timestamp, Timestamp):
                                        temp_timestamp = temp_timestamp.to_pydatetime()
                                except:
                                    pass
                                if hasattr(temp_timestamp, 'to_pydatetime'):
                                    temp_timestamp = temp_timestamp.to_pydatetime()
                                temp_age = end_time - temp_timestamp
            
            # Fetch pressure
            if pressure_tag_id:
                df = self.db_handler.getDataframeBetween(start_time, end_time, [pressure_tag_id])
                if df is not None and not df.empty:
                    if pressure_tag in df.columns:
                        # Get non-null values
                        pressure_series = df[pressure_tag].dropna()
                        if not pressure_series.empty:
                            pressure_value = pressure_series.iloc[-1]
                            # Get timestamp of this value
                            if 'datetime' in df.columns:
                                pressure_timestamp = df.loc[pressure_series.index[-1], 'datetime']
                                try:
                                    from pandas import Timestamp
                                    if isinstance(pressure_timestamp, Timestamp):
                                        pressure_timestamp = pressure_timestamp.to_pydatetime()
                                except:
                                    pass
                                if hasattr(pressure_timestamp, 'to_pydatetime'):
                                    pressure_timestamp = pressure_timestamp.to_pydatetime()
                                pressure_age = end_time - pressure_timestamp
            
            return temp_value, temp_age, pressure_value, pressure_age
        except Exception as e:
            print(f"Error fetching temperature/pressure from database: {e}")
            return None, None, None, None
    
    def calculate_flow_conversion(self):
        """Calculate LPM at actual conditions from SLPM"""
        try:
            # Get SLPM input
            slpm = float(self.slpm_input.get())
            
            # Get temperature (from database or manual)
            if self.use_db_conv.get() and self.db_available:
                temp_c, temp_age, _, _ = self.fetch_temp_pressure_from_db()
                if temp_c is None:
                    self.lpm_output.set("Error: Could not fetch temperature from database")
                    return
                if temp_age is not None and temp_age.total_seconds() > 60:
                    age_str = self.format_age(temp_age)
                    self.temp_db_value.set(f"{temp_c:.2f} °C")
                    self.temp_age_label.config(text=f"(Age: {age_str})")
                else:
                    self.temp_db_value.set(f"{temp_c:.2f} °C")
                    self.temp_age_label.config(text="")
            else:
                temp_c = float(self.temp_manual.get())
                self.temp_db_value.set("---")
                self.temp_age_label.config(text="")
            
            # Get pressure (from database or manual)
            if self.use_db_conv.get() and self.db_available:
                _, _, pressure_bar_gauge, pressure_age = self.fetch_temp_pressure_from_db()
                if pressure_bar_gauge is None:
                    self.lpm_output.set("Error: Could not fetch pressure from database")
                    return
                if pressure_age is not None and pressure_age.total_seconds() > 60:
                    age_str = self.format_age(pressure_age)
                    self.pressure_db_value.set(f"{pressure_bar_gauge:.2f} bar gauge")
                    self.pressure_age_label.config(text=f"(Age: {age_str})")
                else:
                    self.pressure_db_value.set(f"{pressure_bar_gauge:.2f} bar gauge")
                    self.pressure_age_label.config(text="")
            else:
                pressure_bar_gauge = float(self.pressure_manual.get())
                self.pressure_db_value.set("---")
                self.pressure_age_label.config(text="")
            
            # STP conditions: 0°C, 0 bar gauge
            T_STP_K = 0.0 + 273.15  # 273.15 K
            P_STP_bar_abs = 1.01325  # 1 atm = 1.01325 bar absolute (0 bar gauge)
            
            # Actual conditions
            T_actual_K = temp_c + 273.15
            P_actual_bar_abs = pressure_bar_gauge + 1.01325  # Convert gauge to absolute
            
            # Ideal gas law: V_actual = V_STP * (P_STP / P_actual) * (T_actual / T_STP)
            lpm_actual = slpm * (P_STP_bar_abs / P_actual_bar_abs) * (T_actual_K / T_STP_K)
            
            self.lpm_output.set(f"{lpm_actual:.2f} LPM")
            
        except ValueError:
            self.lpm_output.set("Error: Invalid SLPM input")
        except Exception as e:
            self.lpm_output.set(f"Error: {str(e)}")
    
    def convert_gas_from_db(self):
        """Convert selected gas from database LPM to SLPM"""
        try:
            # Get tags for selected gas
            gas = self.selected_gas.get()
            temp_tag, pressure_tag, flow_tag, gas_name = self.get_gas_tags(gas)
            
            # Fetch gas LPM from database
            flow_tag_id = self.get_tag_id_from_name(flow_tag)
            if flow_tag_id is None:
                self.gas_slpm_output.set("Error: Tag not found")
                return
            
            # Get recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=2)
            
            df = self.db_handler.getDataframeBetween(start_time, end_time, [flow_tag_id])
            if df is None or df.empty:
                self.gas_slpm_output.set("Error: No data from database")
                return
            
            if flow_tag not in df.columns:
                self.gas_slpm_output.set("Error: Tag column not found")
                return
            
            gas_lpm = df[flow_tag].dropna().iloc[-1] if not df[flow_tag].dropna().empty else None
            if gas_lpm is None:
                self.gas_slpm_output.set("Error: No valid value")
                return
            
            # Update display
            self.gas_lpm_db.set(f"{gas_lpm:.2f} LPM")
            
            # Store data point for plotting
            now = datetime.now()
            with self.update_lock:
                if gas == "CO2":
                    self.co2_flow_history.append((now, gas_lpm))
                elif gas == "N2":
                    self.n2_flow_history.append((now, gas_lpm))
                elif gas == "H2":
                    self.h2_flow_history.append((now, gas_lpm))
            # Update plot
            self.update_flow_plot()
            
            # Get temperature (from database or manual)
            if self.use_db_conv.get() and self.db_available:
                temp_c, _, _, _ = self.fetch_temp_pressure_from_db()
                if temp_c is None:
                    self.gas_slpm_output.set("Error: Could not fetch temperature from database")
                    return
            else:
                temp_c = float(self.temp_manual.get())
            
            # Get pressure (from database or manual)
            if self.use_db_conv.get() and self.db_available:
                _, _, pressure_bar_gauge, _ = self.fetch_temp_pressure_from_db()
                if pressure_bar_gauge is None:
                    self.gas_slpm_output.set("Error: Could not fetch pressure from database")
                    return
            else:
                pressure_bar_gauge = float(self.pressure_manual.get())
            
            # STP conditions: 20°C, 0 bar gauge
            T_STP_K = 20.0 + 273.15  # 293.15 K
            P_STP_bar_abs = 1.01325  # 1 atm = 1.01325 bar absolute
            
            # Actual conditions
            T_actual_K = temp_c + 273.15
            P_actual_bar_abs = pressure_bar_gauge + 1.01325  # Convert gauge to absolute
            
            # Convert from LPM at actual to SLPM: V_STP = V_actual * (P_actual / P_STP) * (T_STP / T_actual)
            gas_slpm = gas_lpm * (P_actual_bar_abs / P_STP_bar_abs) * (T_STP_K / T_actual_K)
            
            self.gas_slpm_output.set(f"{gas_slpm:.2f} SLPM")
            
        except Exception as e:
            self.gas_slpm_output.set(f"Error: {str(e)}")
    
    def update_flow_plot(self):
        """Update the flow rate plot for the selected gas - fetches from database"""
        gas = self.selected_gas.get()
        _, _, _, gas_name = self.get_gas_tags(gas)
        flow_tag_id = self.get_gas_flow_tag_id(gas)
        
        # Clear and redraw plot
        self.flow_ax.clear()
        
        # Check if flow tag is available
        if flow_tag_id is None:
            # Show warning message for N2 (or other gases without flow tags)
            if gas == "N2":
                self.flow_ax.text(0.5, 0.5, "N2 LPM flow not retrievable!", 
                                 transform=self.flow_ax.transAxes,
                                 fontsize=14, ha='center', va='center',
                                 color='red', weight='bold')
            else:
                self.flow_ax.text(0.5, 0.5, f"{gas_name} flow rate not available", 
                                 transform=self.flow_ax.transAxes,
                                 fontsize=12, ha='center', va='center',
                                 color='orange')
        # Fetch data from database
        elif self.db_available:
            try:
                end_time = datetime.now()
                
                # Calculate start time based on user selection
                if self.flow_time_range_mode.get() == "last":
                    try:
                        time_value = float(self.flow_time_value.get())
                        time_unit = self.flow_time_unit.get()
                        if time_unit == "seconds":
                            start_time = end_time - timedelta(seconds=time_value)
                        elif time_unit == "minutes":
                            start_time = end_time - timedelta(minutes=time_value)
                        elif time_unit == "hours":
                            start_time = end_time - timedelta(hours=time_value)
                        else:
                            start_time = end_time - timedelta(minutes=5)  # Default
                    except ValueError:
                        start_time = end_time - timedelta(minutes=5)  # Default if invalid
                else:
                    start_time = end_time - timedelta(minutes=5)  # Default
                
                df = self.db_handler.getDataframeBetween(start_time, end_time, [flow_tag_id])
                if df is not None and not df.empty:
                    # The database handler maps tag IDs to tag names, so columns are named with tag names
                    # We need to find the tag name that corresponds to this tag ID
                    # tag_id_dict structure is {tag_id: tag_name}
                    tag_name = None
                    if hasattr(self, 'tag_id_dict') and self.tag_id_dict:
                        # Direct lookup: tag_id_dict[tag_id] should give us the tag name
                        if flow_tag_id in self.tag_id_dict:
                            tag_name = self.tag_id_dict[flow_tag_id]
                    
                    flow_col = None
                    # First try to find the tag name in columns
                    if tag_name and tag_name in df.columns:
                        flow_col = tag_name
                    else:
                        # If tag name not found, try getting it directly from the handler's dictionary
                        if hasattr(self.db_handler, 'getTagIDDictionary'):
                            tag_dict = self.db_handler.getTagIDDictionary()
                            if tag_dict and flow_tag_id in tag_dict:
                                handler_tag_name = tag_dict[flow_tag_id]
                                if handler_tag_name in df.columns:
                                    flow_col = handler_tag_name
                        
                        # Fallback: try tag ID as string or number
                        if flow_col is None:
                            tag_id_str = str(flow_tag_id)
                            if tag_id_str in df.columns:
                                flow_col = tag_id_str
                            elif flow_tag_id in df.columns:
                                flow_col = flow_tag_id
                    
                    if flow_col and flow_col in df.columns:
                        # Get non-null values
                        flow_series = df[flow_col].dropna()
                        if not flow_series.empty:
                            # Get timestamps - database handler returns 'DateTime' column
                            datetime_col = 'DateTime' if 'DateTime' in df.columns else 'datetime'
                            if datetime_col in df.columns:
                                times = []
                                values = []
                                for idx in flow_series.index:
                                    timestamp = df.loc[idx, datetime_col]
                                    try:
                                        from pandas import Timestamp
                                        if isinstance(timestamp, Timestamp):
                                            timestamp = timestamp.to_pydatetime()
                                    except:
                                        pass
                                    if hasattr(timestamp, 'to_pydatetime'):
                                        timestamp = timestamp.to_pydatetime()
                                    times.append(timestamp)
                                    values.append(flow_series.loc[idx])
                                
                                if times and values:
                                    self.flow_ax.plot(times, values, 'b-', linewidth=1.5, label=f'{gas_name} Flow Rate')
                                    self.flow_ax.legend(fontsize=8)
                                    if len(values) > 1:
                                        self.flow_ax.set_ylim(min(values) * 0.95, max(values) * 1.05)
                                    else:
                                        self.flow_ax.set_ylim(values[0] * 0.95, values[0] * 1.05)
                                    self.flow_ax.set_xlim(start_time, end_time)
                            else:
                                print(f"Warning: No datetime column found in dataframe. Columns: {df.columns.tolist()}")
                        else:
                            print(f"Warning: No non-null values found for tag ID {flow_tag_id}")
                    else:
                        print(f"Warning: Tag ID {flow_tag_id} not found in dataframe columns: {df.columns.tolist()}")
                else:
                    if df is None or df.empty:
                        print(f"Warning: No data returned from database for tag ID {flow_tag_id}")
            except Exception as e:
                print(f"Error updating flow plot: {e}")
                import traceback
                traceback.print_exc()
        
        self.flow_ax.set_xlabel("Time", fontsize=9)
        self.flow_ax.set_ylabel(f"{gas_name} Flow Rate (LPM)", fontsize=9)
        self.flow_ax.grid(True, alpha=0.3)
        self.flow_fig.tight_layout()
        self.flow_canvas.draw()
    
    def calculate(self):
        """Calculate and display power required"""
        try:
            # Get temperature inputs (from database or manual)
            if self.use_db_temps.get() and self.db_available:
                side1_temp, side2_temp = self.fetch_temperatures_from_db()
                if side1_temp is None or side2_temp is None:
                    self.power_result.set("Error: Could not fetch temperatures from database")
                    self.warning_result.set("Check tag names and database connection")
                    return
            else:
                side1_temp = float(self.side1_temp.get())
                side2_temp = float(self.side2_temp.get())
            
            # Determine inlet and outlet based on which is higher
            # Higher temp = outlet, lower temp = inlet (gas always heats up)
            inlet_temp = min(side1_temp, side2_temp)
            outlet_temp = max(side1_temp, side2_temp)
            co2_slpm = float(self.co2_flow.get())
            h2_slpm = float(self.h2_flow.get())
            n2_slpm = float(self.n2_flow.get())
            co2_conversion = float(self.co2_conversion.get())
            heat_loss = float(self.heat_loss.get())
            
            # Validate inputs
            if co2_conversion < 0 or co2_conversion > 100:
                self.power_result.set("Error: CO₂ conversion must be 0-100%")
                self.warning_result.set("")
                return
            
            if heat_loss < 0:
                self.power_result.set("Error: Heat loss must be >= 0")
                self.warning_result.set("")
                return
            
            # Check for insufficient H2
            co2_mol_per_sec = self.calculator.slpm_to_mol_per_sec(co2_slpm)
            h2_mol_per_sec = self.calculator.slpm_to_mol_per_sec(h2_slpm)
            co2_converted_mol = co2_mol_per_sec * (co2_conversion / 100.0)
            
            if co2_converted_mol > h2_mol_per_sec:
                self.warning_result.set(
                    f"Warning: Insufficient H₂. Need {co2_converted_mol:.3f} mol/s, "
                    f"have {h2_mol_per_sec:.3f} mol/s. Conversion limited to available H₂."
                )
            else:
                self.warning_result.set("")
            
            # Calculate power
            power = self.calculator.calculate_power(
                inlet_temp, outlet_temp,
                co2_slpm, h2_slpm, n2_slpm, co2_conversion
            )
            
            # Add ambient heat loss
            total_power = power + heat_loss
            
            # Calculate +25% power
            power_plus_25 = total_power * 1.25
            
            # Store data point if not in live update mode (manual calculation)
            if not self.live_update_active:
                now = datetime.now()
                with self.update_lock:
                    # Store: (datetime, base_power_without_heat_loss, total_power, power_plus_25)
                    self.power_history.append((now, power, total_power, power_plus_25))
                    self.temp_history.append((now, side1_temp, side2_temp))
                # Update plots
                self.update_plots()
            
            # Display result
            if total_power >= 1000:
                power_kw = total_power / 1000
                self.power_result.set(f"{power_kw:.2f} kW ({total_power:.1f} W)")
            else:
                self.power_result.set(f"{total_power:.1f} W")
            
            if power_plus_25 >= 1000:
                power_plus25_kw = power_plus_25 / 1000
                self.power_plus25_result.set(f"{power_plus25_kw:.2f} kW ({power_plus_25:.1f} W)")
            else:
                self.power_plus25_result.set(f"{power_plus_25:.1f} W")
                
        except ValueError as e:
            self.power_result.set(f"Error: Invalid input - {str(e)}")
            self.power_plus25_result.set("---")
            self.warning_result.set("")
        except Exception as e:
            self.power_result.set(f"Error: {str(e)}")
            self.power_plus25_result.set("---")
            self.warning_result.set("")


def main():
    root = tk.Tk()
    app = PowerCalculatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

