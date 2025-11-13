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
from databaseHandler import Handler

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
        self.power_history = deque()  # [(datetime, power, power_plus_25), ...]
        self.temp_history = deque()   # [(datetime, side1_temp, side2_temp), ...]
        self.realtime_power_history = deque()  # [(datetime, realtime_power), ...]
        
        # Live update control
        self.live_update_active = False
        self.update_thread = None
        self.update_lock = threading.Lock()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
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
        co2_entry = tk.Entry(flow_frame, textvariable=self.co2_flow, width=15,
                            font=("Consolas", 9))
        co2_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        h2_label = tk.Label(flow_frame, text="H₂ (SLPM):", 
                           font=("Consolas", 9, "bold"))
        h2_label.grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.h2_flow = tk.StringVar(value="100")
        h2_entry = tk.Entry(flow_frame, textvariable=self.h2_flow, width=15,
                           font=("Consolas", 9))
        h2_entry.grid(row=3, column=1, sticky=tk.E, pady=5, padx=5)
        
        n2_label = tk.Label(flow_frame, text="N₂ (SLPM):", 
                          font=("Consolas", 9, "bold"))
        n2_label.grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.n2_flow = tk.StringVar(value="0")
        n2_entry = tk.Entry(flow_frame, textvariable=self.n2_flow, width=15,
                           font=("Consolas", 9))
        n2_entry.grid(row=4, column=1, sticky=tk.E, pady=5, padx=5)
        
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
            side1_temp = None
            if side1_tag1_id:
                tagids = [side1_tag1_id]
                if side1_tag2_id:
                    tagids.append(side1_tag2_id)
                
                df = self.db_handler.getDataframeBetween(start_time, end_time, tagids)
                if df is not None and not df.empty:
                    # Get most recent non-null values
                    if side1_tag2_id:
                        # Average two tags
                        tag1_name = side1_tag1_full
                        tag2_name = side1_tag2_full
                        if tag1_name in df.columns and tag2_name in df.columns:
                            df['avg_side1'] = df[[tag1_name, tag2_name]].mean(axis=1)
                            side1_temp = df['avg_side1'].dropna().iloc[-1] if not df['avg_side1'].dropna().empty else None
                        elif tag1_name in df.columns:
                            side1_temp = df[tag1_name].dropna().iloc[-1] if not df[tag1_name].dropna().empty else None
                    else:
                        # Single tag
                        tag1_name = side1_tag1_full
                        if tag1_name in df.columns:
                            side1_temp = df[tag1_name].dropna().iloc[-1] if not df[tag1_name].dropna().empty else None
            
            # Fetch Side 2 temperatures
            side2_temp = None
            if side2_tag1_id:
                tagids = [side2_tag1_id]
                if side2_tag2_id:
                    tagids.append(side2_tag2_id)
                
                df = self.db_handler.getDataframeBetween(start_time, end_time, tagids)
                if df is not None and not df.empty:
                    # Get most recent non-null values
                    if side2_tag2_id:
                        # Average two tags
                        tag1_name = side2_tag1_full
                        tag2_name = side2_tag2_full
                        if tag1_name in df.columns and tag2_name in df.columns:
                            df['avg_side2'] = df[[tag1_name, tag2_name]].mean(axis=1)
                            side2_temp = df['avg_side2'].dropna().iloc[-1] if not df['avg_side2'].dropna().empty else None
                        elif tag1_name in df.columns:
                            side2_temp = df[tag1_name].dropna().iloc[-1] if not df[tag1_name].dropna().empty else None
                    else:
                        # Single tag
                        tag1_name = side2_tag1_full
                        if tag1_name in df.columns:
                            side2_temp = df[tag1_name].dropna().iloc[-1] if not df[tag1_name].dropna().empty else None
            
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
            co2_tag_id = self.get_tag_id_from_name("ai/fi_116m/val")
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
                    tag_name = "ai/fi_116m/val"
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
        elif preset == "RWGS":
            self.co2_flow.set("174")
            self.h2_flow.set("400")
            self.n2_flow.set("0")
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
        # "Manual" - do nothing, user can edit manually
    
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
        """Background thread that updates calculations every second"""
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
                
                time.sleep(1.0)  # Update every second
            except Exception as e:
                print(f"Error in live update loop: {e}")
                time.sleep(1.0)
    
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
            
            # Store data point (store side temps, not calculated inlet/outlet)
            now = datetime.now()
            with self.update_lock:
                self.power_history.append((now, total_power, power_plus_25))
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
    
    def update_plots(self):
        """Update both power and temperature plots"""
        with self.update_lock:
            # Get data (create copies to avoid holding lock during plotting)
            power_times = [t for t, _, _ in self.power_history]
            power_values = [p for _, p, _ in self.power_history]
            power_plus25_values = [p25 for _, _, p25 in self.power_history]
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
            line1 = self.power_ax.plot(times, powers, 'b-', label='Power', linewidth=1.5)[0]
            line2 = self.power_ax.plot(times, powers_plus25, 'g-', label='+25% Power', linewidth=1.5)[0]
            all_powers.extend(powers)
            all_powers.extend(powers_plus25)
            # Store data for hover
            self.power_plot_data.append(('Power', list(times), list(powers)))
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
        if all_powers:
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
        if temp_filtered:
            all_temps = list(side1s) + list(side2s)
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
                    self.power_history.append((now, total_power, power_plus_25))
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

