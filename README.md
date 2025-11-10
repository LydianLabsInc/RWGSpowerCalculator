# Gas Mixture Power Calculator

A Python GUI application that estimates the power required to heat up a gas mixture with CO₂ conversion reaction.

## Features

- Calculate power required to heat gas mixtures (CO₂, H₂, N₂)
- Account for CO₂ conversion reaction: CO₂ + H₂ → CO + H₂O
- Uses NIST thermodynamic data via CoolProp library
- Optional molar ratio tying for flow rates
- Reference state: STP (Standard Temperature and Pressure, 0°C, 1 atm)

## Installation

1. Install Python 3.7 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python power_calculator.py
```

### Input Parameters

- **Inlet Temperature**: Temperature of incoming gas mixture (°C)
- **Outlet Temperature**: Desired outlet temperature (°C)
- **Volumetric Flow Rates**: Flow rates in SLPM (Standard Liters Per Minute)
  - CO₂ (SLPM)
  - H₂ (SLPM)
  - N₂ (SLPM)
- **Molar Ratios** (Optional): Check to tie flow rates together based on ratios
- **CO₂ Conversion**: Percentage of CO₂ moles converted to CO (0-100%)

### Calculation Details

The calculator accounts for:
1. Sensible heat to raise all gases from inlet to outlet temperature
2. Reaction enthalpy for CO₂ + H₂ → CO + H₂O at outlet temperature
3. Heat of formation changes (using NIST standard values)
4. The number of H₂ moles converted equals the number of CO₂ moles converted

### Output

- **Power Required**: Displayed in Watts (or kW for values ≥ 1000 W)

## Technical Notes

- Uses CoolProp library for NIST thermodynamic data
- Enthalpy calculations use STP (0°C, 1 atm) as reference state
- Reaction: CO₂ + H₂ → CO + H₂O (reverse water-gas shift)
- Heat of formation values from NIST:
  - CO₂: -393.5 kJ/mol
  - CO: -110.5 kJ/mol
  - H₂O: -241.8 kJ/mol
  - H₂: 0 kJ/mol (reference)
  - N₂: 0 kJ/mol (reference)

