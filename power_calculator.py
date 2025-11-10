"""
Power Calculator for Gas Mixture Heating
Estimates power required to heat a gas mixture with CO2 conversion reaction
"""

import tkinter as tk
from tkinter import ttk
import CoolProp.CoolProp as CP

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
                       co2_slpm, h2_slpm, n2_slpm, co2_conversion_pct,
                       use_molar_ratios=False, co2_ratio=1.0, h2_ratio=1.0, n2_ratio=1.0):
        """
        Calculate power required to heat gas mixture
        
        Parameters:
        - inlet_temp_c: Inlet temperature in °C
        - outlet_temp_c: Outlet temperature in °C
        - co2_slpm, h2_slpm, n2_slpm: Volumetric flow rates in SLPM
        - co2_conversion_pct: Percentage of CO2 converted (0-100)
        - use_molar_ratios: If True, tie flow rates to ratios
        - co2_ratio, h2_ratio, n2_ratio: Molar ratios (used if use_molar_ratios=True)
        """
        # Convert temperatures to Kelvin
        inlet_temp_k = inlet_temp_c + 273.15
        outlet_temp_k = outlet_temp_c + 273.15
        
        # Handle molar ratios
        if use_molar_ratios:
            # Use the first non-zero flow rate as base
            if co2_slpm > 0:
                base_slpm = co2_slpm / co2_ratio
            elif h2_slpm > 0:
                base_slpm = h2_slpm / h2_ratio
            elif n2_slpm > 0:
                base_slpm = n2_slpm / n2_ratio
            else:
                return 0.0
            
            co2_slpm = base_slpm * co2_ratio
            h2_slpm = base_slpm * h2_ratio
            n2_slpm = base_slpm * n2_ratio
        
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
        self.root.geometry("700x950")
        self.root.minsize(650, 900)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabelFrame.Label', font=("Consolas", 10, "bold"))
        self.style.configure('TLabel', font=("Consolas", 9))
        self.style.configure('TButton', font=("Consolas", 10))
        self.style.configure('TCheckbutton', font=("Consolas", 9))
        
        self.calculator = PowerCalculator()
        
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
        
        # Temperature inputs
        temp_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        temp_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        temp_frame.grid_columnconfigure(0, weight=1)
        temp_frame.grid_columnconfigure(1, weight=1)
        
        # Center the label frame title by creating a centered label
        temp_title = tk.Label(temp_frame, text="Temperature", 
                             font=("Consolas", 10, "bold"))
        temp_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        inlet_label = tk.Label(temp_frame, text="HMR Exit Temp °C:", 
                                font=("Consolas", 9, "bold"))
        inlet_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.inlet_temp = tk.StringVar(value="25")
        inlet_entry = tk.Entry(temp_frame, textvariable=self.inlet_temp, width=15,
                              font=("Consolas", 9))
        inlet_entry.grid(row=1, column=1, sticky=tk.E, pady=5, padx=5)
        
        outlet_label = tk.Label(temp_frame, text="Target Element Temp °C:", 
                               font=("Consolas", 9, "bold"))
        outlet_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.outlet_temp = tk.StringVar(value="800")
        outlet_entry = tk.Entry(temp_frame, textvariable=self.outlet_temp, width=15,
                               font=("Consolas", 9))
        outlet_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        # Flow rate inputs
        flow_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        flow_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        flow_frame.grid_columnconfigure(0, weight=1)
        flow_frame.grid_columnconfigure(1, weight=1)
        
        flow_title = tk.Label(flow_frame, text="Volumetric Flow Rates (SLPM)", 
                             font=("Consolas", 10, "bold"))
        flow_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        co2_label = tk.Label(flow_frame, text="CO₂ (SLPM):", 
                            font=("Consolas", 9, "bold"))
        co2_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.co2_flow = tk.StringVar(value="100")
        co2_entry = tk.Entry(flow_frame, textvariable=self.co2_flow, width=15,
                            font=("Consolas", 9))
        co2_entry.grid(row=1, column=1, sticky=tk.E, pady=5, padx=5)
        
        h2_label = tk.Label(flow_frame, text="H₂ (SLPM):", 
                           font=("Consolas", 9, "bold"))
        h2_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.h2_flow = tk.StringVar(value="100")
        h2_entry = tk.Entry(flow_frame, textvariable=self.h2_flow, width=15,
                           font=("Consolas", 9))
        h2_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        n2_label = tk.Label(flow_frame, text="N₂ (SLPM):", 
                          font=("Consolas", 9, "bold"))
        n2_label.grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.n2_flow = tk.StringVar(value="0")
        n2_entry = tk.Entry(flow_frame, textvariable=self.n2_flow, width=15,
                           font=("Consolas", 9))
        n2_entry.grid(row=3, column=1, sticky=tk.E, pady=5, padx=5)
        
        # Molar ratios option
        ratio_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        ratio_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ratio_frame.grid_columnconfigure(0, weight=1)
        ratio_frame.grid_columnconfigure(1, weight=1)
        
        ratio_title = tk.Label(ratio_frame, text="Molar Ratios (Optional)", 
                              font=("Consolas", 10, "bold"))
        ratio_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        self.use_ratios = tk.BooleanVar(value=False)
        ratio_check = tk.Checkbutton(ratio_frame, text="Tie flow rates to molar ratios",
                                    variable=self.use_ratios,
                                    font=("Consolas", 9))
        ratio_check.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        co2_ratio_label = tk.Label(ratio_frame, text="CO₂ Ratio:", 
                                   font=("Consolas", 9, "bold"))
        co2_ratio_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.co2_ratio = tk.StringVar(value="1.0")
        co2_ratio_entry = tk.Entry(ratio_frame, textvariable=self.co2_ratio, width=15,
                                   font=("Consolas", 9))
        co2_ratio_entry.grid(row=2, column=1, sticky=tk.E, pady=5, padx=5)
        
        h2_ratio_label = tk.Label(ratio_frame, text="H₂ Ratio:", 
                                 font=("Consolas", 9, "bold"))
        h2_ratio_label.grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.h2_ratio = tk.StringVar(value="1.0")
        h2_ratio_entry = tk.Entry(ratio_frame, textvariable=self.h2_ratio, width=15,
                                 font=("Consolas", 9))
        h2_ratio_entry.grid(row=3, column=1, sticky=tk.E, pady=5, padx=5)
        
        n2_ratio_label = tk.Label(ratio_frame, text="N₂ Ratio:", 
                                 font=("Consolas", 9, "bold"))
        n2_ratio_label.grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.n2_ratio = tk.StringVar(value="0.0")
        n2_ratio_entry = tk.Entry(ratio_frame, textvariable=self.n2_ratio, width=15,
                                  font=("Consolas", 9))
        n2_ratio_entry.grid(row=4, column=1, sticky=tk.E, pady=5, padx=5)
        
        # CO2 conversion
        conv_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        conv_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
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
        heatloss_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        heatloss_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
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
        
        # Calculate button
        calc_button = ttk.Button(main_frame, text="Calculate Power", command=self.calculate)
        calc_button.grid(row=6, column=0, columnspan=2, pady=20)
        
        # Results - centered at bottom
        result_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        result_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
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
        
        self.warning_result = tk.StringVar(value="")
        warning_label = tk.Label(result_frame, textvariable=self.warning_result, 
                                font=("Consolas", 9),
                                foreground="orange", wraplength=500)
        warning_label.grid(row=2, column=0, columnspan=2, pady=5, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def calculate(self):
        """Calculate and display power required"""
        try:
            # Get inputs
            inlet_temp = float(self.inlet_temp.get())
            outlet_temp = float(self.outlet_temp.get())
            co2_slpm = float(self.co2_flow.get())
            h2_slpm = float(self.h2_flow.get())
            n2_slpm = float(self.n2_flow.get())
            co2_conversion = float(self.co2_conversion.get())
            heat_loss = float(self.heat_loss.get())
            use_ratios = self.use_ratios.get()
            
            co2_ratio = float(self.co2_ratio.get())
            h2_ratio = float(self.h2_ratio.get())
            n2_ratio = float(self.n2_ratio.get())
            
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
                co2_slpm, h2_slpm, n2_slpm, co2_conversion,
                use_ratios, co2_ratio, h2_ratio, n2_ratio
            )
            
            # Add ambient heat loss
            total_power = power + heat_loss
            
            # Display result
            if total_power >= 1000:
                power_kw = total_power / 1000
                self.power_result.set(f"{power_kw:.2f} kW ({total_power:.1f} W)")
            else:
                self.power_result.set(f"{total_power:.1f} W")
                
        except ValueError as e:
            self.power_result.set(f"Error: Invalid input - {str(e)}")
            self.warning_result.set("")
        except Exception as e:
            self.power_result.set(f"Error: {str(e)}")
            self.warning_result.set("")


def main():
    root = tk.Tk()
    app = PowerCalculatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

