/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
// -*- C++ -*-
// ----------------------------------------------------------------------
// HEP coherent system of Units
//
// This file has been provided to CLHEP by Geant4 (simulation toolkit for HEP).
//
// The basic units are :
//  		millimeter              (millimeter)
// 		nanosecond              (nanosecond)
// 		Mega electron Volt      (MeV)
// 		positron charge         (eplus)
// 		degree Kelvin           (kelvin)
//              the amount of substance (mole)
//              luminous intensity      (candela)
// 		radian                  (radian)
//              steradian               (steradian)
//
// Below is a non exhaustive list of derived and pratical units
// (i.e. mostly the SI units).
// You can add your own units.
//
// The SI numerical value of the positron charge is defined here,
// as it is needed for conversion factor : positron charge = e_SI (coulomb)
//
// The others physical constants are defined in the header file :
//			PhysicalConstants.h
//
// Authors: M.Maire, S.Giani
//
// History:
//
// 06.02.96   Created.
// 28.03.96   Added miscellaneous constants.
// 05.12.97   E.Tcherniaev: Redefined pascal (to avoid warnings on WinNT)
// 20.05.98   names: meter, second, gram, radian, degree
//            (from Brian.Lasiuk@yale.edu (STAR)). Added luminous units.
// 05.08.98   angstrom, picobarn, microsecond, picosecond, petaelectronvolt
// 01.03.01   parsec
// 28.04.06   import from CLHEP to GaudiKernel -- HD
// 11.05.06   Rename pascal to Pa to avoid warnings on Windows - MC

#ifndef GAUDI_SYSTEM_OF_UNITS_H
#define GAUDI_SYSTEM_OF_UNITS_H

namespace Gaudi {
  namespace Units {

    //
    // Length [L]
    //
    constexpr float millimeter = 1.f;
    constexpr float millimeter2 = millimeter * millimeter;
    constexpr float millimeter3 = millimeter * millimeter * millimeter;

    constexpr float centimeter = 10.f * millimeter;
    constexpr float centimeter2 = centimeter * centimeter;
    constexpr float centimeter3 = centimeter * centimeter * centimeter;

    constexpr float meter = 1000.f * millimeter;
    constexpr float meter2 = meter * meter;
    constexpr float meter3 = meter * meter * meter;

    constexpr float kilometer = 1000.f * meter;
    constexpr float kilometer2 = kilometer * kilometer;
    constexpr float kilometer3 = kilometer * kilometer * kilometer;

    constexpr float parsec = 3.0856775807e+16f * meter;

    constexpr float micrometer = 1.e-6f * meter;
    constexpr float nanometer = 1.e-9f * meter;
    constexpr float angstrom = 1.e-10f * meter;
    constexpr float fermi = 1.e-15f * meter;

    constexpr float barn = 1.e-28f * meter2;
    constexpr float millibarn = 1.e-3f * barn;
    constexpr float microbarn = 1.e-6f * barn;
    constexpr float nanobarn = 1.e-9f * barn;
    constexpr float picobarn = 1.e-12f * barn;

    // symbols
    constexpr float nm = nanometer;
    constexpr float um = micrometer;

    constexpr float mm = millimeter;
    constexpr float mm2 = millimeter2;
    constexpr float mm3 = millimeter3;

    constexpr float cm = centimeter;
    constexpr float cm2 = centimeter2;
    constexpr float cm3 = centimeter3;

    constexpr float m = meter;
    constexpr float m2 = meter2;
    constexpr float m3 = meter3;

    constexpr float km = kilometer;
    constexpr float km2 = kilometer2;
    constexpr float km3 = kilometer3;

    constexpr float pc = parsec;

    //
    // Angle
    //
    constexpr float radian = 1.f;
    constexpr float milliradian = 1.e-3f * radian;
    constexpr float degree = (3.14159265358979323846f / 180.0f) * radian;

    constexpr float steradian = 1.f;

    // symbols
    constexpr float rad = radian;
    constexpr float mrad = milliradian;
    constexpr float sr = steradian;
    constexpr float deg = degree;

    //
    // Time [T]
    //
    constexpr float nanosecond = 1.f;
    constexpr float second = 1.e+9f * nanosecond;
    constexpr float millisecond = 1.e-3f * second;
    constexpr float microsecond = 1.e-6f * second;
    constexpr float picosecond = 1.e-12f * second;
    constexpr float femtosecond = 1.e-15f * second;

    constexpr float hertz = 1.f / second;
    constexpr float kilohertz = 1.e+3f * hertz;
    constexpr float megahertz = 1.e+6f * hertz;

    // symbols
    constexpr float ns = nanosecond;
    constexpr float s = second;
    constexpr float ms = millisecond;

    //
    // Electric charge [Q]
    //
    constexpr float eplus = 1.f;            // positron charge
    constexpr float e_SI = 1.60217733e-19f; // positron charge in coulomb
    constexpr float coulomb = eplus / e_SI; // coulomb = 6.24150 e+18 * eplus

    //
    // Energy [E]
    //
    constexpr float megaelectronvolt = 1.f;
    constexpr float electronvolt = 1.e-6f * megaelectronvolt;
    constexpr float kiloelectronvolt = 1.e-3f * megaelectronvolt;
    constexpr float gigaelectronvolt = 1.e+3f * megaelectronvolt;
    constexpr float teraelectronvolt = 1.e+6f * megaelectronvolt;
    constexpr float petaelectronvolt = 1.e+9f * megaelectronvolt;

    constexpr float joule = electronvolt / e_SI; // joule = 6.24150 e+12 * MeV

    // symbols
    constexpr float MeV = megaelectronvolt;
    constexpr float eV = electronvolt;
    constexpr float keV = kiloelectronvolt;
    constexpr float GeV = gigaelectronvolt;
    constexpr float TeV = teraelectronvolt;
    constexpr float PeV = petaelectronvolt;

    //
    // Mass [E][T^2][L^-2]
    //
    constexpr float kilogram = joule * second * second / (meter * meter);
    constexpr float gram = 1.e-3f * kilogram;
    constexpr float milligram = 1.e-3f * gram;

    // symbols
    constexpr float kg = kilogram;
    constexpr float g = gram;
    constexpr float mg = milligram;

    //
    // Power [E][T^-1]
    //
    constexpr float watt = joule / second; // watt = 6.24150 e+3 * MeV/ns

    //
    // Force [E][L^-1]
    //
    constexpr float newton = joule / meter; // newton = 6.24150 e+9 * MeV/mm

    //
    // Pressure [E][L^-3]
    //
    constexpr float Pa = newton / m2;         // pascal = 6.24150 e+3 * MeV/mm3
    constexpr float bar = 100000 * Pa;        // bar    = 6.24150 e+8 * MeV/mm3
    constexpr float atmosphere = 101325 * Pa; // atm    = 6.32420 e+8 * MeV/mm3

    //
    // Electric current [Q][T^-1]
    //
    constexpr float ampere = coulomb / second; // ampere = 6.24150 e+9 * eplus/ns
    constexpr float milliampere = 1.e-3f * ampere;
    constexpr float microampere = 1.e-6f * ampere;
    constexpr float nanoampere = 1.e-9f * ampere;

    //
    // Electric potential [E][Q^-1]
    //
    constexpr float megavolt = megaelectronvolt / eplus;
    constexpr float kilovolt = 1.e-3f * megavolt;
    constexpr float volt = 1.e-6f * megavolt;

    //
    // Electric resistance [E][T][Q^-2]
    //
    constexpr float ohm = volt / ampere; // ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

    //
    // Electric capacitance [Q^2][E^-1]
    //
    constexpr float farad = coulomb / volt; // farad = 6.24150e+24 * eplus/Megavolt
    constexpr float millifarad = 1.e-3f * farad;
    constexpr float microfarad = 1.e-6f * farad;
    constexpr float nanofarad = 1.e-9f * farad;
    constexpr float picofarad = 1.e-12f * farad;

    //
    // Magnetic Flux [T][E][Q^-1]
    //
    constexpr float weber = volt * second; // weber = 1000*megavolt*ns

    //
    // Magnetic Field [T][E][Q^-1][L^-2]
    //
    constexpr float tesla = volt * second / meter2; // tesla =0.001*megavolt*ns/mm2

    constexpr float gauss = 1.e-4f * tesla;
    constexpr float kilogauss = 1.e-1f * tesla;

    //
    // Inductance [T^2][E][Q^-2]
    //
    constexpr float henry = weber / ampere; // henry = 1.60217e-7*MeV*(ns/eplus)**2

    //
    // Temperature
    //
    constexpr float kelvin = 1.f;

    //
    // Amount of substance
    //
    constexpr float mole = 1.f;

    //
    // Activity [T^-1]
    //
    constexpr float becquerel = 1.f / second;
    constexpr float curie = 3.7e+10f * becquerel;

    //
    // Absorbed dose [L^2][T^-2]
    //
    constexpr float gray = joule / kilogram;

    //
    // Luminous intensity [I]
    //
    constexpr float candela = 1.f;

    //
    // Luminous flux [I]
    //
    constexpr float lumen = candela * steradian;

    //
    // Illuminance [I][L^-2]
    //
    constexpr float lux = lumen / meter2;

    //
    // Miscellaneous
    //
    constexpr float perCent = 0.01f;
    constexpr float perThousand = 0.001f;
    constexpr float perMillion = 0.000001f;

  } // namespace Units
} // namespace Gaudi

#endif /* GAUDI_SYSTEM_OF_UNITS_H */
