'use client';

import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  Play,
  Square,
  RotateCcw,
  Volume2,
  VolumeX,
  Sliders,
  Activity,
  Cpu,
  Globe,
  BookOpen,
  Download,
  AlertOctagon,
  ShieldAlert,
  Sparkles,
  FlaskConical,
  Filter,
  Layers,
  Droplets,
  Syringe,
  Maximize2,
  Minimize2,
  PanelRightClose,
  PanelRightOpen,
  Scan,
  Check,
  X,
  Wheat,
  ExternalLink,
  type LucideIcon,
} from 'lucide-react';
import { Manrope, IBM_Plex_Mono } from 'next/font/google';

const fontUi = Manrope({ subsets: ['latin'], display: 'swap', variable: '--font-ui' });
const fontData = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
  variable: '--font-data',
});

/* ============================================================================
 * 0. SMALL UTILITIES
 * ========================================================================== */
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const logistic = (x: number) => 1 / (1 + Math.exp(-x));
const decimalsOf = (step: number) => (String(step).split('.')[1] ?? '').length;

/* ============================================================================
 * 1. DOMAIN CONSTANTS
 *    NOTE: every coefficient below is a *placeholder calibration* for teaching.
 *    Replace with literature / plant data before quoting numbers anywhere.
 * ========================================================================== */
const BIO = {
  muMax: 0.048, //          1/h
  Ks: 0.85, //              g/L glucose
  Yield_XS: 0.62, //        g DCW / g glucose
  qO2: 18, //               %air-sat/h per g/L DCW (calibrated: DO ≈ 40 % at 8 g/L, 320 rpm)
  Ko2: 2, //                % air-sat, oxygen Monod constant
  lacInhib: 6, //           g/L lactate at which growth is halved
  xMax: 9, //              g/L DCW crowding ceiling (space / unmodelled nutrients)
  maintenance: 0.03, //    g glucose / g DCW / h
  kDeath: 0.003, //         1/h baseline cell death
  qP: 0.002, //             g antigen / g DCW / h
  optimalTemp: 37.0,
  tempWidth: 3.8,
  optimalPh: 7.15,
  phWidth: 0.55,
  criticalShear: 1.5, //    Pa
  shearK: 3.4e-6,
  shearExp: 1.96,
  aggOnsetTemp: 37.8, //    °C
  cellsPerGram: 3.3, //     10^6 cells/mL per g/L DCW (CHO, approx.)
  batchVolumeL: 2000,
  targetCultureH: 336, //   14 days
} as const;

const LIMITS = {
  tempLow: 36.0,
  tempHigh: 38.0,
  rpmSafeMax: 650,
  doWarn: 25,
  doCrit: 10,
  phLow: 6.8,
  phHigh: 7.4,
  glucoseLow: 1.0,
} as const;

/** Release specification (illustrative). */
const SPEC = { minPurity: 95, maxHcpPpm: 100, maxDnaNgPerDose: 10, minLrv: 4.0 } as const;

const SIM = {
  dt: 0.08, //               simulated hours per integration step
  hoursPerSecond: 0.8, //    simulated hours per real second at 1×
  maxStepsPerFrame: 60,
  publishMs: 100,
  histEveryUp: 2, //         h between upstream chart points
  histEveryDs: 0.12, //      h between downstream chart points
  histMax: 400,
} as const;

const MIN_HARVEST_TITER = 0.1; // g/L
const UFDF_AREA_M2 = 40;

/* ============================================================================
 * 2. UPSTREAM MODEL (fed-batch CHO, RK4)
 * ========================================================================== */
const UP_KEYS = [
  'biomass',
  'substrate',
  'dissolvedOxygen',
  'monomerPurity',
  'brothPh',
  'lactate',
  'titer',
  'debris',
] as const;
type UpKey = (typeof UP_KEYS)[number];
type UpState = Record<UpKey, number>;

interface UpInputs {
  rpm: number;
  temperature: number;
  feedRate: number;
  spargeRate: number;
  phSetpoint: number;
}

const INITIAL_UP: UpState = {
  biomass: 0.55, //          g/L DCW
  substrate: 24.0, //        g/L
  dissolvedOxygen: 90.0, //  % air-sat
  monomerPurity: 99.4, //    %
  brothPh: 7.15,
  lactate: 0.12, //          g/L
  titer: 0, //               g/L antigen
  debris: 0.012, //          g/L dead-cell mass
};

const INITIAL_UP_INPUTS: UpInputs = {
  rpm: 320,
  temperature: 37.0,
  feedRate: 6.5,
  spargeRate: 0.12,
  phSetpoint: 7.15,
};

const UP_CLAMP: Record<UpKey, [number, number]> = {
  biomass: [0.01, Infinity],
  substrate: [0, Infinity],
  dissolvedOxygen: [0, 100],
  monomerPurity: [0, 100],
  brothPh: [6.0, 8.2],
  lactate: [0, Infinity],
  titer: [0, Infinity],
  debris: [0, Infinity],
};

/** Shear is algebraic in rpm, so it is derived, never stored. */
const shearAt = (rpm: number) => BIO.shearK * Math.pow(rpm, BIO.shearExp);

function upDerivatives(s: UpState, u: UpInputs): UpState {
  const tempInh = Math.max(0, 1 - ((u.temperature - BIO.optimalTemp) / BIO.tempWidth) ** 2);
  const phInh = Math.max(0, 1 - ((s.brothPh - BIO.optimalPh) / BIO.phWidth) ** 2);
  const oxInh = s.dissolvedOxygen / (s.dissolvedOxygen + BIO.Ko2);
  const lacInh = 1 / (1 + (s.lactate / BIO.lacInhib) ** 2);
  const crowd = Math.max(0, 1 - s.biomass / BIO.xMax);
  const sat = s.substrate / (BIO.Ks + s.substrate);
  const mu = BIO.muMax * sat * tempInh * phInh * oxInh * lacInh * crowd;

  const overShear = Math.max(0, shearAt(u.rpm) - BIO.criticalShear);
  const loss = BIO.kDeath + overShear * 0.22;

  const kLa = 0.0022 * Math.pow(u.rpm, 1.18) + u.spargeRate * 3.4;
  const dLactate = 0.004 * s.biomass * (s.substrate / (0.5 + s.substrate)) - 0.008 * s.lactate;
  const aggT = u.temperature > BIO.aggOnsetTemp ? 0.12 * Math.pow(u.temperature - BIO.aggOnsetTemp, 1.8) : 0;

  return {
    biomass: (mu - loss) * s.biomass,
    substrate: -(mu / BIO.Yield_XS + BIO.maintenance * sat) * s.biomass + u.feedRate * 0.045,
    dissolvedOxygen: kLa * (100 - s.dissolvedOxygen) - BIO.qO2 * s.biomass,
    monomerPurity: -(aggT + overShear * 1.85),
    brothPh: (u.phSetpoint - s.brothPh) * 0.45 - dLactate * 0.04,
    lactate: dLactate,
    titer: BIO.qP * s.biomass * phInh * oxInh - 0.0002 * s.titer,
    debris: loss * s.biomass,
  };
}

const upAdvance = (s: UpState, k: UpState, h: number): UpState => {
  const out = {} as UpState;
  for (const key of UP_KEYS) out[key] = clamp(s[key] + h * k[key], UP_CLAMP[key][0], UP_CLAMP[key][1]);
  return out;
};

function stepUpstream(s: UpState, u: UpInputs, dt: number): UpState {
  const k1 = upDerivatives(s, u);
  const k2 = upDerivatives(upAdvance(s, k1, dt / 2), u);
  const k3 = upDerivatives(upAdvance(s, k2, dt / 2), u);
  const k4 = upDerivatives(upAdvance(s, k3, dt), u);
  const out = {} as UpState;
  for (const key of UP_KEYS) {
    out[key] = clamp(
      s[key] + (dt / 6) * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]),
      UP_CLAMP[key][0],
      UP_CLAMP[key][1],
    );
  }
  return out;
}

/* ============================================================================
 * 3. DOWNSTREAM MODEL (harvest → clarification → capture → viral inactivation
 *    → polishing → UF/DF → sterile filtration & fill)
 * ========================================================================== */
const DS_ORDER = ['CLARIFY', 'CAPTURE', 'VIRAL', 'POLISH', 'UFDF', 'FILL'] as const;
type DsKey = (typeof DS_ORDER)[number];

interface DsInputs {
  fluxLMH: number; //      clarification flux (L/m²/h)
  areaM2: number; //       depth-filter area (m²)
  resMin: number; //       capture residence time (min)
  loadDensity: number; //  capture load (g antigen / L resin)
  elutionPh: number;
  holdPh: number; //       viral-inactivation hold pH
  holdMin: number;
  holdTemp: number;
  polCond: number; //      polishing load conductivity (mS/cm)
  polLoad: number; //      polishing load (g/L resin)
  tmpBar: number; //       UF/DF transmembrane pressure
  crossflow: number; //    L/min/m²
  dvCount: number; //      diavolumes
  targetConc: number; //   g/L drug-substance concentration
  filtBar: number; //      sterile filter differential pressure
  fillSpeed: number; //    vials/min
  doseUg: number; //       µg antigen per dose
}

const INITIAL_DS_INPUTS: DsInputs = {
  fluxLMH: 100,
  areaM2: 36,
  resMin: 4,
  loadDensity: 35,
  elutionPh: 3.5,
  holdPh: 3.6,
  holdMin: 60,
  holdTemp: 20,
  polCond: 6.5,
  polLoad: 80,
  tmpBar: 1.2,
  crossflow: 5,
  dvCount: 7,
  targetConc: 20,
  filtBar: 1.5,
  fillSpeed: 300,
  doseUg: 30,
};

interface DsLive {
  pressure: number;
  limit: number; //        0 = no limit applies
  turbidity: number;
  breakthrough: number; // %
  ph: number;
  conductivity: number;
  loadFrac: number;
  volumeFrac: number;
  label: string;
}

interface DsState {
  status: 'IDLE' | 'RUNNING' | 'COMPLETE' | 'FAILED';
  stage: number; //        index into DS_ORDER
  progress: number; //     0..1 within stage
  volume: number; //       L
  product: number; //      g antigen
  startProduct: number;
  purity: number; //       % monomer
  hcp: number; //          ppm (ng/mg)
  dna: number; //          ng/mg
  debris: number; //       g/L carried from harvest
  lrv: number; //          viral-inactivation log reduction
  live: DsLive;
  vials: { total: number; filled: number; rejected: number };
  scratch: { v0: number; p0: number; a: number; dv: number; phase: number };
  recovery: number[]; //   cumulative recovery (%) after each finished stage
  failure: string | null;
}

const EMPTY_LIVE: DsLive = {
  pressure: 0,
  limit: 0,
  turbidity: 0,
  breakthrough: 0,
  ph: 7.0,
  conductivity: 0,
  loadFrac: 0,
  volumeFrac: 1,
  label: 'READY',
};

function beginStage(s: DsState, inp: DsInputs) {
  const sc = { v0: s.volume, p0: s.product, a: 0, dv: 0, phase: 0 };
  if (DS_ORDER[s.stage] === 'UFDF') {
    sc.a = Math.min(sc.v0, Math.max(20, (s.product * 0.97) / inp.targetConc));
    sc.phase = sc.v0 > sc.a + 1e-6 ? 0 : 1;
  }
  s.scratch = sc;
  s.live = { ...EMPTY_LIVE };
}

function makeDownstream(up: UpState, inp: DsInputs): DsState {
  const product = up.titer * BIO.batchVolumeL;
  const s: DsState = {
    status: 'IDLE',
    stage: 0,
    progress: 0,
    volume: BIO.batchVolumeL,
    product,
    startProduct: Math.max(product, 1e-9),
    purity: up.monomerPurity,
    hcp: 4e5 * (1 + 0.2 * up.debris),
    dna: 2e7 * (1 + 0.3 * up.debris),
    debris: up.debris,
    lrv: 0,
    live: { ...EMPTY_LIVE },
    vials: { total: 0, filled: 0, rejected: 0 },
    scratch: { v0: 0, p0: 0, a: 0, dv: 0, phase: 0 },
    recovery: [],
    failure: null,
  };
  s.scratch.v0 = s.volume;
  s.scratch.p0 = s.product;
  beginStage(s, inp);
  return s;
}

function stepDownstream(prev: DsState, inp: DsInputs, dt: number): { state: DsState; boundary: boolean } {
  if (prev.status !== 'RUNNING') return { state: prev, boundary: false };

  const s: DsState = {
    ...prev,
    live: { ...prev.live },
    vials: { ...prev.vials },
    scratch: { ...prev.scratch },
  };
  const sc = s.scratch;
  const key: DsKey = DS_ORDER[s.stage];
  let dp = 0;
  let newProg = s.progress;
  let failure: string | null = null;

  switch (key) {
    case 'CLARIFY': {
      const dur = sc.v0 / (inp.fluxLMH * inp.areaM2);
      dp = Math.min(1 - s.progress, dt / dur);
      newProg = s.progress + dp;
      const cap = 240 / (1 + 0.3 * s.debris); // L/m² filter capacity
      const ratio = (newProg * sc.v0) / inp.areaM2 / cap;
      const yieldT = clamp(0.95 - (0.05 * Math.max(0, inp.fluxLMH - 100)) / 100, 0.8, 0.95);
      s.product *= Math.pow(yieldT, dp);
      s.hcp *= Math.pow(0.85, dp);
      s.dna *= Math.pow(10, -0.5 * dp);
      s.volume = sc.v0 * (1 + 0.05 * newProg);
      s.live.pressure = 0.15 + 0.25 * (inp.fluxLMH / 100) + 1.8 * ratio * ratio;
      s.live.limit = 2.0;
      s.live.turbidity = 3 * (inp.fluxLMH / 100) * (1 + 2 * ratio * ratio);
      s.live.loadFrac = clamp(ratio, 0, 1.5);
      s.live.label = 'DEPTH FILTRATION';
      s.live.ph = 7.1;
      s.live.conductivity = 14;
      if (s.live.pressure > s.live.limit) {
        failure = `Depth filter blocked (ΔP ${s.live.pressure.toFixed(2)} bar > 2.0). Lower the flux or add filter area, then retry from harvest.`;
      }
      break;
    }

    case 'CAPTURE': {
      const resin = Math.max(1, sc.p0 / inp.loadDensity);
      const dbc10 = 60 * (1 - Math.exp(-inp.resMin / 1.5)); // g/L at 10 % breakthrough
      const rMax = inp.loadDensity / dbc10;
      const loadH = (sc.v0 * inp.resMin) / (60 * resin);
      const dur = loadH + 1.5;
      const loadShare = loadH / dur;
      dp = Math.min(1 - s.progress, dt / dur);
      newProg = s.progress + dp;

      const K = 10;
      const R0 = 1.22;
      const loadFrac = Math.min(1, newProg / loadShare);
      const bt = logistic(K * (rMax * loadFrac - R0));
      const avgBt = (Math.log(1 + Math.exp(K * (rMax - R0))) - Math.log(1 + Math.exp(-K * R0))) / (K * rMax);
      const elRec = 0.97 / (1 + Math.exp((inp.elutionPh - 3.9) * 6));
      const yieldT = (1 - avgBt) * elRec;
      const hcpLrv = 2.8 - (0.9 * Math.max(0, 3.6 - inp.elutionPh)) / 0.6;

      s.product *= Math.pow(yieldT, dp);
      s.hcp *= Math.pow(10, -hcpLrv * dp);
      s.dna *= Math.pow(10, -3 * dp);
      s.purity -= Math.max(0, 3.4 - inp.elutionPh) * 2 * dp;
      const vEnd = Math.max(20, (sc.p0 * yieldT) / 12);
      s.volume = sc.v0 + (vEnd - sc.v0) * newProg;

      const washEnd = loadShare + (1 - loadShare) * 0.4;
      const inLoad = newProg < loadShare;
      const inWash = !inLoad && newProg < washEnd;
      s.live.label = inLoad ? 'LOAD' : inWash ? 'WASH' : 'ELUTE';
      s.live.ph = inLoad ? 7.4 : inWash ? 7.0 : inp.elutionPh;
      s.live.conductivity = inLoad ? 14 : inWash ? 12 : 4;
      s.live.breakthrough = inLoad ? bt * 100 : 0;
      s.live.loadFrac = loadFrac;
      s.live.pressure = 0.4 + 3.2 / inp.resMin;
      s.live.limit = 3.0;
      break;
    }

    case 'VIRAL': {
      const holdH = inp.holdMin / 60;
      const dur = 0.5 + holdH;
      dp = Math.min(1 - s.progress, dt / dur);
      newProg = s.progress + dp;
      const a = 0.25 / dur;
      const b = a + holdH / dur;
      const holdFrac = clamp((newProg - a) / (b - a), 0, 1);
      const tFac = Math.pow(2, (inp.holdTemp - 20) / 10);
      const k = 0.021 * Math.pow(10, (4.0 - inp.holdPh) * 1.6) * tFac; // LRV per minute
      s.lrv = Math.min(6.5, k * inp.holdMin) * holdFrac;
      const agg = 0.9 * Math.pow(Math.max(0, 4.0 - inp.holdPh), 2) * holdH * tFac;
      s.purity -= agg * dp;
      s.product *= Math.pow(0.985, dp);
      s.volume = sc.v0 * (1 + 0.05 * newProg);
      s.live.ph =
        newProg < a
          ? 7.0 + (inp.holdPh - 7.0) * (newProg / a)
          : newProg < b
            ? inp.holdPh
            : inp.holdPh + (7.0 - inp.holdPh) * ((newProg - b) / (1 - b));
      s.live.label = newProg < a ? 'ACIDIFY' : newProg < b ? 'HOLD' : 'NEUTRALISE';
      s.live.conductivity = 6;
      s.live.pressure = 0.2;
      s.live.loadFrac = newProg;
      break;
    }

    case 'POLISH': {
      dp = Math.min(1 - s.progress, dt / 3.0);
      newProg = s.progress + dp;
      const fCond = 1 / (1 + Math.exp((inp.polCond - 9) * 0.9));
      const fLoad = 1 / (1 + Math.exp((inp.polLoad - 140) / 18));
      const f = fCond * fLoad;
      const yieldT = clamp(
        0.97 - 0.12 * Math.max(0, (5 - inp.polCond) / 2) - 0.08 * Math.max(0, (inp.polLoad - 130) / 100),
        0.5,
        0.97,
      );
      s.hcp *= Math.pow(10, -3.0 * f * dp);
      s.dna *= Math.pow(10, -4.0 * f * dp);
      s.product *= Math.pow(yieldT, dp);
      s.purity += Math.max(0, 99.8 - s.purity) * 0.6 * dp;
      s.volume = sc.v0 * (1 + 0.1 * newProg);
      s.live.label = 'FLOW-THROUGH';
      s.live.ph = 7.2;
      s.live.conductivity = inp.polCond;
      s.live.pressure = 0.3 + 0.5 * (inp.polLoad / 100);
      s.live.loadFrac = newProg;
      break;
    }

    case 'UFDF': {
      const vc = sc.a;
      const cb = s.product / Math.max(s.volume, 1);
      const flux =
        9 * Math.sqrt(inp.crossflow / 5) * Math.log(250 / Math.max(cb, 0.5)) * (1 - Math.exp(-inp.tmpBar / 0.8));
      const perm = Math.max(0, flux) * UFDF_AREA_M2 * dt;
      if (sc.phase === 0) {
        s.volume = Math.max(vc, s.volume - perm);
        newProg = 0.5 * ((sc.v0 - s.volume) / Math.max(sc.v0 - vc, 1e-6));
        if (s.volume <= vc + 1e-6) sc.phase = 1;
      } else {
        sc.dv += perm / vc;
        newProg = 0.5 + 0.5 * Math.min(1, sc.dv / inp.dvCount);
      }
      newProg = clamp(Math.max(s.progress, newProg), 0, 1);
      dp = newProg - s.progress;
      s.product *= Math.exp(-0.002 * (1 + (inp.tmpBar / 1.2) ** 2 * (inp.crossflow / 5) ** 2) * dt);
      s.purity -= 0.05 * ((inp.tmpBar * inp.crossflow) / 6) ** 2 * dt;
      s.hcp *= Math.pow(0.8, dp);
      s.live.pressure = inp.tmpBar;
      s.live.limit = 2.0;
      s.live.volumeFrac = s.volume / Math.max(sc.v0, 1);
      s.live.label = sc.phase === 0 ? 'CONCENTRATE' : 'DIAFILTER';
      s.live.conductivity = 3 + 12 * Math.exp(-sc.dv);
      s.live.ph = 6.8;
      s.live.loadFrac = newProg;
      break;
    }

    case 'FILL': {
      const dur = 3.0 * (300 / inp.fillSpeed); // campaign time compressed for the simulator
      dp = Math.min(1 - s.progress, dt / dur);
      newProg = s.progress + dp;
      const filtShare = 0.25;
      const pressure = inp.filtBar * (0.6 + 0.5 * newProg * newProg);
      const fillFrac = clamp((newProg - filtShare) / (1 - filtShare), 0, 1);
      const rejectRate = 0.005 + 0.04 * Math.pow(Math.max(0, (inp.fillSpeed - 420) / 180), 2);
      s.product *= Math.pow(0.985, dp);
      s.vials.total = (s.product * 1e6) / inp.doseUg / 10; // 10-dose vials
      s.vials.filled = s.vials.total * fillFrac;
      s.vials.rejected = s.vials.filled * rejectRate;
      s.live.pressure = pressure;
      s.live.limit = 2.8;
      s.live.volumeFrac = 1 - fillFrac;
      s.live.loadFrac = newProg;
      s.live.label = newProg < filtShare ? 'STERILE FILTRATION' : 'FILLING';
      s.live.ph = 6.8;
      s.live.conductivity = 3;
      if (pressure > s.live.limit) {
        failure = `Sterilising filter ΔP ${pressure.toFixed(2)} bar exceeds 2.8 bar: membrane integrity at risk. Lower the filtration pressure and retry.`;
      }
      break;
    }
  }

  s.progress = newProg;

  if (failure) {
    s.status = 'FAILED';
    s.failure = failure;
    return { state: s, boundary: true };
  }

  if (s.progress >= 1 - 1e-9) {
    if (key === 'VIRAL' && s.lrv < SPEC.minLrv) {
      s.status = 'FAILED';
      s.failure = `Insufficient viral clearance: LRV ${s.lrv.toFixed(1)} < ${SPEC.minLrv.toFixed(1)}. Lower the hold pH, extend the hold or raise the temperature.`;
      return { state: s, boundary: true };
    }
    s.recovery = [...s.recovery, (s.product / s.startProduct) * 100];
    if (s.stage === DS_ORDER.length - 1) {
      s.status = 'COMPLETE';
    } else {
      s.stage += 1;
      s.progress = 0;
      beginStage(s, inp);
    }
    return { state: s, boundary: true };
  }
  return { state: s, boundary: false };
}

interface ReleaseCheck {
  name: string;
  value: string;
  spec: string;
  pass: boolean;
}

function releaseChecks(ds: DsState, inp: DsInputs): ReleaseCheck[] {
  const dnaPerDose = ds.dna * (inp.doseUg / 1000);
  return [
    {
      name: 'Antigen purity (SEC monomer)',
      value: `${ds.purity.toFixed(2)} %`,
      spec: `≥ ${SPEC.minPurity} %`,
      pass: ds.purity >= SPEC.minPurity,
    },
    {
      name: 'Host-cell protein',
      value: `${ds.hcp.toFixed(1)} ppm`,
      spec: `≤ ${SPEC.maxHcpPpm} ppm`,
      pass: ds.hcp <= SPEC.maxHcpPpm,
    },
    {
      name: 'Residual DNA',
      value: `${dnaPerDose.toFixed(3)} ng/dose`,
      spec: `≤ ${SPEC.maxDnaNgPerDose} ng/dose`,
      pass: dnaPerDose <= SPEC.maxDnaNgPerDose,
    },
    {
      name: 'Viral inactivation',
      value: `${ds.lrv.toFixed(1)} LRV`,
      spec: `≥ ${SPEC.minLrv.toFixed(1)} LRV`,
      pass: ds.lrv >= SPEC.minLrv,
    },
  ];
}

/* ============================================================================
 * 4. ALARMS
 * ========================================================================== */
type Phase = 'UPSTREAM' | 'DOWNSTREAM';
type Severity = 'warning' | 'critical';
interface Alarm {
  id: string;
  severity: Severity;
  text: string;
}

function evaluateAlarms(phase: Phase, up: UpState, upIn: UpInputs, ds: DsState | null, dsIn: DsInputs): Alarm[] {
  const out: Alarm[] = [];
  const add = (id: string, severity: Severity, text: string) => out.push({ id, severity, text });

  if (phase === 'UPSTREAM') {
    const tau = shearAt(upIn.rpm);
    if (tau > BIO.criticalShear) add('shear', 'critical', `Shear ${tau.toFixed(2)} Pa exceeds ${BIO.criticalShear} Pa: cell lysis`);
    if (upIn.temperature > LIMITS.tempHigh) add('temp-hi', 'critical', `Temperature ${upIn.temperature.toFixed(1)} °C above ${LIMITS.tempHigh}: antigen aggregation`);
    else if (upIn.temperature < LIMITS.tempLow) add('temp-lo', 'warning', `Temperature ${upIn.temperature.toFixed(1)} °C below ${LIMITS.tempLow}: slow growth`);
    if (up.dissolvedOxygen < LIMITS.doCrit) add('do', 'critical', `dO₂ ${up.dissolvedOxygen.toFixed(0)} %: oxygen starvation`);
    else if (up.dissolvedOxygen < LIMITS.doWarn) add('do', 'warning', `dO₂ ${up.dissolvedOxygen.toFixed(0)} %: oxygen-limited`);
    if (up.monomerPurity < SPEC.minPurity) add('purity', 'critical', `Monomer purity ${up.monomerPurity.toFixed(1)} % below ${SPEC.minPurity} %`);
    if (up.substrate < LIMITS.glucoseLow) add('glc', 'warning', `Glucose ${up.substrate.toFixed(1)} g/L: depleted, raise the feed`);
    if (up.brothPh < LIMITS.phLow || up.brothPh > LIMITS.phHigh) add('ph', 'warning', `Broth pH ${up.brothPh.toFixed(2)} outside ${LIMITS.phLow}–${LIMITS.phHigh}`);
  } else if (ds) {
    if (ds.failure) {
      add('ds-fail', 'critical', ds.failure);
    } else {
      const key = DS_ORDER[ds.stage];
      if (ds.live.limit > 0 && ds.live.pressure > 0.8 * ds.live.limit) {
        add('ds-pressure', 'warning', `${key} pressure ${ds.live.pressure.toFixed(2)} bar approaching ${ds.live.limit.toFixed(1)} bar limit`);
      }
      if (key === 'CAPTURE' && ds.live.breakthrough > 10) add('ds-bt', 'warning', `Capture breakthrough ${ds.live.breakthrough.toFixed(0)} %: product lost to flow-through`);
      if (ds.purity < SPEC.minPurity) add('ds-purity', 'critical', `Antigen purity ${ds.purity.toFixed(1)} % below ${SPEC.minPurity} %`);
    }
  }
  void dsIn;
  return out;
}

/* ============================================================================
 * 5. STAGE & PARAMETER CATALOGUE (single source of truth for UI + 3D)
 * ========================================================================== */
interface StageDef {
  key: 'UPSTREAM' | DsKey;
  label: string;
  short: string;
  Icon: LucideIcon;
  blurb: string;
}

const STAGES: StageDef[] = [
  { key: 'UPSTREAM', label: 'Upstream bioreactor', short: 'Upstream', Icon: FlaskConical, blurb: 'Fed-batch culture. Balance growth, oxygen and shear to build titer without damaging the antigen.' },
  { key: 'CLARIFY', label: 'Harvest & clarification', short: 'Clarify', Icon: Filter, blurb: 'Depth filtration removes cells and debris. Flux and area set the pressure the filter sees.' },
  { key: 'CAPTURE', label: 'Capture chromatography', short: 'Capture', Icon: Layers, blurb: 'Bind-and-elute step. Overloading causes breakthrough; harsh elution raises recovery but aggregates antigen.' },
  { key: 'VIRAL', label: 'Viral inactivation', short: 'Virus', Icon: ShieldAlert, blurb: 'Low-pH hold. Lower pH and longer holds clear more virus but cost purity.' },
  { key: 'POLISH', label: 'Polishing', short: 'Polish', Icon: Sparkles, blurb: 'Flow-through step that strips residual host-cell protein, DNA and aggregates.' },
  { key: 'UFDF', label: 'UF/DF (TFF)', short: 'UF/DF', Icon: Droplets, blurb: 'Concentrate then buffer-exchange. High TMP and crossflow raise flux but shear the antigen.' },
  { key: 'FILL', label: 'Sterile filtration & fill', short: 'Fill', Icon: Syringe, blurb: 'Final 0.22 µm filtration, then vial filling. Speed trades throughput against reject rate.' },
];

type DialColor = 'cyan' | 'emerald' | 'amber' | 'rose';
interface ParamDef {
  group: 'up' | 'ds';
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  safeMin?: number;
  safeMax?: number;
  color: DialColor;
}

const PARAMS: ParamDef[][] = [
  [
    { group: 'up', key: 'rpm', label: 'Agitation', unit: 'rpm', min: 100, max: 800, step: 10, safeMin: 200, safeMax: LIMITS.rpmSafeMax, color: 'cyan' },
    { group: 'up', key: 'temperature', label: 'Temperature', unit: '°C', min: 32, max: 42, step: 0.1, safeMin: LIMITS.tempLow, safeMax: LIMITS.tempHigh, color: 'emerald' },
    { group: 'up', key: 'feedRate', label: 'Glucose feed', unit: 'mL/min', min: 0, max: 20, step: 0.5, safeMin: 2, safeMax: 12, color: 'cyan' },
    { group: 'up', key: 'spargeRate', label: 'Sparging', unit: 'vvm', min: 0.02, max: 0.5, step: 0.01, safeMin: 0.05, safeMax: 0.35, color: 'amber' },
    { group: 'up', key: 'phSetpoint', label: 'pH setpoint', unit: 'pH', min: 6.6, max: 7.8, step: 0.05, safeMin: 6.9, safeMax: 7.4, color: 'emerald' },
  ],
  [
    { group: 'ds', key: 'fluxLMH', label: 'Filter flux', unit: 'LMH', min: 40, max: 250, step: 5, safeMin: 60, safeMax: 150, color: 'cyan' },
    { group: 'ds', key: 'areaM2', label: 'Filter area', unit: 'm²', min: 10, max: 50, step: 2, safeMin: 20, safeMax: 50, color: 'emerald' },
  ],
  [
    { group: 'ds', key: 'resMin', label: 'Residence time', unit: 'min', min: 2, max: 8, step: 0.5, safeMin: 3, safeMax: 6, color: 'cyan' },
    { group: 'ds', key: 'loadDensity', label: 'Resin load', unit: 'g/L', min: 10, max: 60, step: 1, safeMin: 20, safeMax: 45, color: 'amber' },
    { group: 'ds', key: 'elutionPh', label: 'Elution pH', unit: 'pH', min: 3.0, max: 4.2, step: 0.05, safeMin: 3.3, safeMax: 3.8, color: 'emerald' },
  ],
  [
    { group: 'ds', key: 'holdPh', label: 'Hold pH', unit: 'pH', min: 3.3, max: 4.2, step: 0.05, safeMin: 3.4, safeMax: 3.8, color: 'emerald' },
    { group: 'ds', key: 'holdMin', label: 'Hold time', unit: 'min', min: 30, max: 120, step: 5, safeMin: 50, safeMax: 90, color: 'cyan' },
    { group: 'ds', key: 'holdTemp', label: 'Hold temp', unit: '°C', min: 15, max: 30, step: 1, safeMin: 18, safeMax: 24, color: 'amber' },
  ],
  [
    { group: 'ds', key: 'polCond', label: 'Conductivity', unit: 'mS/cm', min: 3, max: 14, step: 0.5, safeMin: 5, safeMax: 8.5, color: 'cyan' },
    { group: 'ds', key: 'polLoad', label: 'Resin load', unit: 'g/L', min: 20, max: 200, step: 5, safeMin: 40, safeMax: 120, color: 'amber' },
  ],
  [
    { group: 'ds', key: 'tmpBar', label: 'TMP', unit: 'bar', min: 0.5, max: 2.5, step: 0.1, safeMin: 0.8, safeMax: 1.6, color: 'cyan' },
    { group: 'ds', key: 'crossflow', label: 'Crossflow', unit: 'L/min/m²', min: 3, max: 10, step: 0.5, safeMin: 4, safeMax: 7, color: 'emerald' },
    { group: 'ds', key: 'dvCount', label: 'Diavolumes', unit: 'DV', min: 3, max: 12, step: 1, safeMin: 6, safeMax: 10, color: 'amber' },
    { group: 'ds', key: 'targetConc', label: 'Target conc.', unit: 'g/L', min: 5, max: 50, step: 1, safeMin: 10, safeMax: 30, color: 'cyan' },
  ],
  [
    { group: 'ds', key: 'filtBar', label: 'Filter ΔP', unit: 'bar', min: 0.5, max: 3.0, step: 0.1, safeMin: 0.8, safeMax: 2.0, color: 'cyan' },
    { group: 'ds', key: 'fillSpeed', label: 'Fill speed', unit: 'vials/min', min: 100, max: 600, step: 10, safeMin: 200, safeMax: 400, color: 'amber' },
    { group: 'ds', key: 'doseUg', label: 'Dose', unit: 'µg', min: 5, max: 100, step: 5, safeMin: 10, safeMax: 60, color: 'emerald' },
  ],
];

/* ---------- teaching scenarios: one-click parameter sets + the lesson they teach ---------- */
interface Scenario {
  label: string;
  tone: 'amber' | 'rose' | 'cyan';
  up?: Partial<UpInputs>;
  ds?: Partial<DsInputs>;
  lesson: string;
}
const SCENARIO_TONE = { amber: 'text-amber-300', rose: 'text-rose-300', cyan: 'text-cyan-300' } as const;

const SCENARIOS: Scenario[][] = [
  [
    { label: 'Thermal trip', tone: 'amber', up: { temperature: 40.8 }, lesson: 'Above 37.8 °C the antigen starts to aggregate and the loss accumulates every hour. Watch purity fall toward the 95 % limit.' },
    { label: 'Shear lysis', tone: 'rose', up: { rpm: 780 }, lesson: 'Past 1.5 Pa cells rupture. Watch viability fall, then notice how much extra debris the depth filter has to handle at harvest.' },
    { label: 'Gas starve', tone: 'cyan', up: { spargeRate: 0.02, rpm: 120 }, lesson: 'Oxygen supply (kLa) drops below cell demand, so dO₂ heads to zero and growth stalls.' },
  ],
  [{ label: 'Undersized filter', tone: 'rose', ds: { fluxLMH: 200, areaM2: 14 }, lesson: 'Too much flux on too little area loads the filter past its capacity. Pressure passes 2 bar and the step fails.' }],
  [
    { label: 'Overloaded column', tone: 'amber', ds: { loadDensity: 60, resMin: 2 }, lesson: 'Loading past the binding capacity at a short residence time lets antigen break through to waste. Recovery drops sharply.' },
    { label: 'Harsh elution', tone: 'cyan', ds: { elutionPh: 3.0 }, lesson: 'Low elution pH recovers more antigen but aggregates some of it and lets more host-cell protein through.' },
  ],
  [{ label: 'Weak viral hold', tone: 'rose', ds: { holdPh: 4.0 }, lesson: 'At pH 4.0 the hold clears far less virus than the 4 log needed. The batch is rejected at the end of the step.' }],
  [{ label: 'Poor polishing', tone: 'amber', ds: { polCond: 13, polLoad: 190 }, lesson: 'The step runs without any alarm, but impurities stay high. The batch only fails at release, which is why in-process checks matter.' }],
  [{ label: 'Harsh TFF', tone: 'rose', ds: { tmpBar: 2.4, crossflow: 9 }, lesson: 'High TMP and crossflow move liquid faster but shear the antigen. Expect product loss and a purity drop.' }],
  [{ label: 'Overpressure', tone: 'rose', ds: { filtBar: 3.0 }, lesson: 'Pushing the sterilising filter past 2.8 bar risks its integrity, so the step is stopped.' }],
];

/* ============================================================================
 * 6. SIMULATION HOOK (fixed-step accumulator, refs for hot state, 10 Hz publish)
 * ========================================================================== */
type HistPt = Record<string, number>;
interface Deviation {
  id: number;
  hours: number;
  phase: Phase;
  severity: 'info' | Severity;
  text: string;
}

interface SimCore {
  phase: Phase;
  up: UpState;
  upHours: number;
  peakShear: number;
  harvestUp: UpState | null;
  ds: DsState | null;
  dsHours: number;
  upHist: HistPt[];
  dsHist: HistPt[];
  nextUpHist: number;
  nextDsHist: number;
  deviations: Deviation[];
  active: Set<string>;
  seq: number;
}

interface Snapshot {
  phase: Phase;
  up: UpState;
  upHours: number;
  peakShear: number;
  ds: DsState | null;
  dsHours: number;
  upHist: HistPt[];
  dsHist: HistPt[];
  deviations: Deviation[];
}

const upPoint = (s: UpState, t: number): HistPt => ({
  t,
  vcd: s.biomass * BIO.cellsPerGram,
  viab: (100 * s.biomass) / (s.biomass + s.debris),
  glc: s.substrate,
  do2: s.dissolvedOxygen,
  purity: s.monomerPurity,
  titer: s.titer,
});

const dsPoint = (d: DsState, t: number): HistPt => ({
  t,
  prog: ((d.stage + d.progress) / DS_ORDER.length) * 100,
  rec: (d.product / d.startProduct) * 100,
  purity: d.purity,
  press: d.live.pressure,
  hcp: Math.log10(Math.max(d.hcp, 1e-3)),
  lrv: d.lrv,
});

const newCore = (): SimCore => ({
  phase: 'UPSTREAM',
  up: { ...INITIAL_UP },
  upHours: 0,
  peakShear: shearAt(INITIAL_UP_INPUTS.rpm),
  harvestUp: null,
  ds: null,
  dsHours: 0,
  upHist: [upPoint(INITIAL_UP, 0)],
  dsHist: [],
  nextUpHist: SIM.histEveryUp,
  nextDsHist: SIM.histEveryDs,
  deviations: [],
  active: new Set(),
  seq: 0,
});

const snapshotOf = (c: SimCore): Snapshot => ({
  phase: c.phase,
  up: c.up,
  upHours: c.upHours,
  peakShear: c.peakShear,
  ds: c.ds,
  dsHours: c.dsHours,
  upHist: c.upHist.slice(),
  dsHist: c.dsHist.slice(),
  deviations: c.deviations.slice(),
});

function useProcessSimulation() {
  const coreRef = useRef<SimCore | null>(null);
  if (!coreRef.current) coreRef.current = newCore();

  const [snap, setSnap] = useState<Snapshot>(() => snapshotOf(coreRef.current!));
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [pauseBetween, setPauseBetween] = useState(false);
  const [upIn, setUpIn] = useState<UpInputs>(INITIAL_UP_INPUTS);
  const [dsIn, setDsIn] = useState<DsInputs>(INITIAL_DS_INPUTS);

  const upInRef = useRef(upIn);
  const dsInRef = useRef(dsIn);
  const speedRef = useRef(speed);
  const pauseRef = useRef(pauseBetween);
  useEffect(() => { upInRef.current = upIn; }, [upIn]);
  useEffect(() => { dsInRef.current = dsIn; }, [dsIn]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { pauseRef.current = pauseBetween; }, [pauseBetween]);

  const publish = useCallback(() => setSnap(snapshotOf(coreRef.current!)), []);

  const note = (c: SimCore, severity: Deviation['severity'], text: string) => {
    const hours = c.phase === 'UPSTREAM' ? c.upHours : c.dsHours;
    c.deviations.push({ id: ++c.seq, hours, phase: c.phase, severity, text });
    if (c.deviations.length > 200) c.deviations.shift();
  };

  const logAlarms = (c: SimCore) => {
    const now = evaluateAlarms(c.phase, c.up, upInRef.current, c.ds, dsInRef.current);
    const ids = new Set(now.map((a) => a.id));
    for (const a of now) if (!c.active.has(a.id)) note(c, a.severity, a.text);
    c.active = ids;
  };

  const trim = (h: HistPt[]) => { if (h.length > SIM.histMax) h.splice(0, h.length - SIM.histMax); };

  /** One fixed integration step. Returns false when the loop should stop. */
  const stepOnce = (): boolean => {
    const c = coreRef.current!;
    if (c.phase === 'UPSTREAM') {
      c.up = stepUpstream(c.up, upInRef.current, SIM.dt);
      c.upHours += SIM.dt;
      c.peakShear = Math.max(c.peakShear, shearAt(upInRef.current.rpm));
      if (c.upHours >= c.nextUpHist) {
        c.upHist.push(upPoint(c.up, c.upHours));
        trim(c.upHist);
        c.nextUpHist += SIM.histEveryUp;
      }
      logAlarms(c);
      if (c.upHours >= BIO.targetCultureH) {
        note(c, 'info', 'Target culture duration reached (14 d). Harvest to continue.');
        return false;
      }
      return true;
    }
    if (!c.ds) return false;
    const { state, boundary } = stepDownstream(c.ds, dsInRef.current, SIM.dt);
    c.ds = state;
    c.dsHours += SIM.dt;
    if (c.dsHours >= c.nextDsHist || boundary) {
      c.dsHist.push(dsPoint(state, c.dsHours));
      trim(c.dsHist);
      c.nextDsHist = c.dsHours + SIM.histEveryDs;
    }
    if (boundary && state.status === 'RUNNING') note(c, 'info', `${STAGES[state.stage].label} complete (recovery ${state.recovery[state.recovery.length - 1].toFixed(1)} %)`);
    if (state.status === 'COMPLETE') note(c, 'info', 'Downstream train complete');
    logAlarms(c);
    if (state.status !== 'RUNNING') return false;
    if (boundary && pauseRef.current) return false;
    return true;
  };

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    let lastPublish = 0;
    let acc = 0;
    const tick = (now: number) => {
      const real = Math.min(0.25, (now - last) / 1000);
      last = now;
      acc += real * SIM.hoursPerSecond * speedRef.current;
      let steps = 0;
      let keepGoing = true;
      while (acc >= SIM.dt && steps < SIM.maxStepsPerFrame && keepGoing) {
        keepGoing = stepOnce();
        acc -= SIM.dt;
        steps++;
      }
      if (steps >= SIM.maxStepsPerFrame) acc = 0;
      if (!keepGoing) {
        setRunning(false);
        publish();
        return;
      }
      if (now - lastPublish >= SIM.publishMs) {
        lastPublish = now;
        publish();
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, publish]);

  const toggleRun = () => {
    const c = coreRef.current!;
    if (running) { setRunning(false); return; }
    if (c.phase === 'DOWNSTREAM') {
      if (!c.ds || c.ds.status === 'COMPLETE' || c.ds.status === 'FAILED') return;
      if (c.ds.status === 'IDLE') c.ds = { ...c.ds, status: 'RUNNING' };
    } else if (c.upHours >= BIO.targetCultureH) {
      return;
    }
    setRunning(true);
    publish();
  };

  const startDownstreamFrom = (c: SimCore, up: UpState) => {
    c.harvestUp = { ...up };
    c.phase = 'DOWNSTREAM';
    c.ds = makeDownstream(up, dsInRef.current);
    c.dsHours = 0;
    c.dsHist = [dsPoint(c.ds, 0)];
    c.nextDsHist = SIM.histEveryDs;
    c.active = new Set();
  };

  const harvest = () => {
    const c = coreRef.current!;
    if (c.phase !== 'UPSTREAM' || c.up.titer < MIN_HARVEST_TITER) return;
    setRunning(false);
    note(c, 'info', `Harvest at ${c.upHours.toFixed(0)} h: titer ${c.up.titer.toFixed(2)} g/L, ${(c.up.titer * BIO.batchVolumeL / 1000).toFixed(2)} kg antigen`);
    startDownstreamFrom(c, c.up);
    publish();
  };

  const retryDownstream = () => {
    const c = coreRef.current!;
    if (!c.harvestUp) return;
    setRunning(false);
    note(c, 'info', 'Downstream retried from harvest');
    startDownstreamFrom(c, c.harvestUp);
    publish();
  };

  const reset = () => {
    setRunning(false);
    coreRef.current = newCore();
    setUpIn(INITIAL_UP_INPUTS);
    setDsIn(INITIAL_DS_INPUTS);
    publish();
  };

  const setParam = useCallback((id: string, value: number) => {
    const [group, key] = id.split(':');
    if (group === 'up') setUpIn((p) => ({ ...p, [key]: value }));
    else setDsIn((p) => ({ ...p, [key]: value }));
  }, []);

  const applyParams = (up?: Partial<UpInputs>, ds?: Partial<DsInputs>) => {
    if (up) setUpIn((p) => ({ ...p, ...up }));
    if (ds) setDsIn((p) => ({ ...p, ...ds }));
  };

  const restoreNominal = (stageIdx: number) => {
    const up: Record<string, number> = {};
    const ds: Record<string, number> = {};
    for (const p of PARAMS[stageIdx]) {
      if (p.group === 'up') up[p.key] = (INITIAL_UP_INPUTS as unknown as Record<string, number>)[p.key];
      else ds[p.key] = (INITIAL_DS_INPUTS as unknown as Record<string, number>)[p.key];
    }
    applyParams(up as Partial<UpInputs>, ds as Partial<DsInputs>);
  };

  /** Demo helper: integrate the culture synchronously (a full 14 d run is only ~4,200 cheap RK4 steps). */
  const fastForward = (hours: number) => {
    const c = coreRef.current!;
    if (c.phase !== 'UPSTREAM') return;
    setRunning(false);
    const target = Math.min(BIO.targetCultureH, c.upHours + hours);
    let guard = 0;
    while (c.upHours < target - 1e-9 && guard++ < 6000) if (!stepOnce()) break;
    publish();
  };

  return {
    snap, running, speed, setSpeed, pauseBetween, setPauseBetween,
    upIn, dsIn, setParam, toggleRun, harvest, retryDownstream, reset, applyParams, restoreNominal, fastForward,
  };
}

/* ============================================================================
 * 7. AUDIO
 * ========================================================================== */
class AudioAlarm {
  private ctx: AudioContext | null = null;
  private osc: OscillatorNode | null = null;
  private muted = false;

  private ensure(): AudioContext | null {
    if (!this.ctx && typeof window !== 'undefined') {
      const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctor();
    }
    if (this.ctx?.state === 'suspended') void this.ctx.resume();
    return this.ctx;
  }

  setMuted(m: boolean) {
    this.muted = m;
    if (m) this.stopAlarm();
  }

  chime(freq = 880, dur = 0.15) {
    if (this.muted) return;
    try {
      const ctx = this.ensure();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + dur);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + dur);
    } catch { /* audio unavailable */ }
  }

  startAlarm() {
    if (this.muted || this.osc) return;
    try {
      const ctx = this.ensure();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(520, ctx.currentTime);
      gain.gain.setValueAtTime(0.05, ctx.currentTime);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      this.osc = osc;
    } catch { /* audio unavailable */ }
  }

  stopAlarm() {
    if (!this.osc) return;
    try { this.osc.stop(); this.osc.disconnect(); } catch { /* already stopped */ }
    this.osc = null;
  }
}
const audio = new AudioAlarm();

/* ============================================================================
 * 8. ROTARY DIAL (pointer events, keyboard, memoised)
 * ========================================================================== */
const DIAL_COLORS: Record<DialColor, string> = {
  cyan: '#06b6d4',
  emerald: '#10b981',
  amber: '#f59e0b',
  rose: '#f43f5e',
};
const DIAL_R = 38;
const DIAL_CIRC = 2 * Math.PI * DIAL_R;
const DIAL_ARC = DIAL_CIRC * (240 / 360);

interface DialProps {
  id: string;
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  safeMin?: number;
  safeMax?: number;
  color: DialColor;
  disabled?: boolean;
  onChange: (id: string, value: number) => void;
}

const RotaryDial = memo(function RotaryDial({ id, label, unit, value, min, max, step, safeMin, safeMax, color, disabled, onChange }: DialProps) {
  const dialRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const decimals = decimalsOf(step);
  const pct = clamp((value - min) / (max - min), 0, 1);
  const pctRef = useRef(pct);
  pctRef.current = pct;

  const out = (safeMin !== undefined && value < safeMin) || (safeMax !== undefined && value > safeMax);
  const hex = out ? '#ef4444' : DIAL_COLORS[color];

  const commit = useCallback(
    (raw: number) => onChange(id, clamp(Number((Math.round(raw / step) * step).toFixed(decimals)), min, max)),
    [id, step, decimals, min, max, onChange],
  );

  const fromPointer = useCallback(
    (cx: number, cy: number) => {
      const el = dialRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      let deg = (Math.atan2(cy - (r.top + r.height / 2), cx - (r.left + r.width / 2)) * 180) / Math.PI + 90;
      if (deg > 180) deg -= 360;
      if (Math.abs(deg) > 120) deg = pctRef.current > 0.5 ? 120 : -120; // dead zone: stick to nearest end
      commit(min + ((deg + 120) / 240) * (max - min));
    },
    [commit, min, max],
  );

  const onKey = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') { e.preventDefault(); commit(value + step); }
    if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') { e.preventDefault(); commit(value - step); }
  };

  return (
    <div className={`flex min-w-0 flex-col items-center rounded-xl border border-slate-800/80 bg-slate-950/70 p-2.5 select-none ${disabled ? 'opacity-45' : ''}`}>
      <div className="flex w-full items-baseline justify-between gap-1 font-mono text-[11px] text-slate-400">
        <span className="truncate font-semibold">{label}</span>
        <span className="shrink-0 text-[10px] text-slate-500">{min}–{max}</span>
      </div>

      <div
        ref={dialRef}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-disabled={disabled}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          dragging.current = true;
          fromPointer(e.clientX, e.clientY);
        }}
        onPointerMove={(e) => { if (dragging.current) fromPointer(e.clientX, e.clientY); }}
        onPointerUp={(e) => { dragging.current = false; e.currentTarget.releasePointerCapture(e.pointerId); }}
        onPointerCancel={() => { dragging.current = false; }}
        className={`relative my-1 aspect-square w-full max-w-28 touch-none outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/70 rounded-full ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <svg className="h-full w-full" viewBox="0 0 100 100" style={{ transform: 'rotate(150deg)' }} aria-hidden>
          <circle cx="50" cy="50" r={DIAL_R} fill="none" stroke="#1e293b" strokeWidth="7" strokeDasharray={`${DIAL_ARC} ${DIAL_CIRC}`} strokeLinecap="round" />
          <circle cx="50" cy="50" r={DIAL_R} fill="none" stroke={hex} strokeWidth="7" strokeDasharray={`${pct * DIAL_ARC} ${DIAL_CIRC}`} strokeLinecap="round" />
        </svg>
        <div
          style={{ transform: `rotate(${-120 + pct * 240}deg)` }}
          className="absolute left-1/2 top-1/2 flex h-[58%] w-[58%] -translate-x-1/2 -translate-y-1/2 items-start justify-center rounded-full border border-slate-700 bg-linear-to-b from-slate-800 to-slate-900 p-1 shadow-xl"
        >
          <div style={{ backgroundColor: hex, boxShadow: `0 0 8px ${hex}` }} className="h-[22%] w-[9%] min-w-0.75 rounded-full" />
        </div>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-sm font-bold leading-none text-white">{value.toFixed(decimals)}</span>
          <span className="mt-0.5 max-w-[60%] truncate font-mono text-[9px] text-slate-400">{unit}</span>
        </div>
      </div>

      <div className="flex w-full items-center gap-1">
        <button disabled={disabled} onClick={() => commit(value - step)} aria-label={`Decrease ${label}`} className="rounded border border-slate-800 bg-slate-900 px-2 font-mono text-xs text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed">−</button>
        <input type="range" disabled={disabled} min={min} max={max} step={step} value={value} onChange={(e) => commit(Number(e.target.value))} aria-label={`${label} slider`} className="h-1 min-w-0 flex-1 cursor-pointer appearance-none rounded-lg bg-slate-800 accent-cyan-400" />
        <button disabled={disabled} onClick={() => commit(value + step)} aria-label={`Increase ${label}`} className="rounded border border-slate-800 bg-slate-900 px-2 font-mono text-xs text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed">+</button>
      </div>
    </div>
  );
});

/* ============================================================================
 * 9. RESPONSIVE CANVAS CHART (ResizeObserver, DPR-aware, hover read-out)
 * ========================================================================== */
interface Channel {
  key: string;
  label: string;
  color: string;
  min: number;
  max: number;
  unit: string;
  digits?: number;
}

const UP_CHART_A: Channel[] = [
  { key: 'vcd', label: 'VCD', color: '#f59e0b', min: 0, max: 40, unit: '×10⁶/mL', digits: 1 },
  { key: 'viab', label: 'Viability', color: '#10b981', min: 50, max: 100, unit: '%', digits: 1 },
  { key: 'glc', label: 'Glucose', color: '#a78bfa', min: 0, max: 30, unit: 'g/L', digits: 1 },
];
const UP_CHART_B: Channel[] = [
  { key: 'do2', label: 'dO₂', color: '#06b6d4', min: 0, max: 100, unit: '%', digits: 0 },
  { key: 'purity', label: 'Purity', color: '#34d399', min: 90, max: 100, unit: '%', digits: 1 },
  { key: 'titer', label: 'Titer', color: '#f472b6', min: 0, max: 5, unit: 'g/L', digits: 2 },
];
const DS_CHART_A: Channel[] = [
  { key: 'prog', label: 'Train progress', color: '#06b6d4', min: 0, max: 100, unit: '%', digits: 0 },
  { key: 'rec', label: 'Recovery', color: '#f59e0b', min: 0, max: 100, unit: '%', digits: 1 },
  { key: 'purity', label: 'Purity', color: '#34d399', min: 80, max: 100, unit: '%', digits: 1 },
];
const DS_CHART_B: Channel[] = [
  { key: 'press', label: 'Pressure', color: '#f43f5e', min: 0, max: 3.5, unit: 'bar', digits: 2 },
  { key: 'hcp', label: 'log₁₀ HCP', color: '#a78bfa', min: -1, max: 6, unit: 'ppm', digits: 1 },
  { key: 'lrv', label: 'Viral LRV', color: '#10b981', min: 0, max: 7, unit: 'log', digits: 1 },
];

const LiveChart = memo(function LiveChart({ title, channels, data }: { title: string; channels: Channel[]; data: HistPt[] }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ w: Math.floor(width), h: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w < 10 || size.h < 10) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const pad = { l: 4, r: 4, t: 6, b: 16 };
    const pw = size.w - pad.l - pad.r;
    const ph = size.h - pad.t - pad.b;

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ph * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + pw, y);
      ctx.stroke();
    }

    if (data.length < 2) return;
    const t0 = data[0].t;
    const t1 = data[data.length - 1].t;
    const span = Math.max(t1 - t0, 1e-6);
    const xOf = (t: number) => pad.l + ((t - t0) / span) * pw;
    const yOf = (v: number, c: Channel) => pad.t + ph - clamp((v - c.min) / (c.max - c.min), 0, 1) * ph;

    ctx.fillStyle = '#64748b';
    ctx.font = `10px ${fontData.style.fontFamily}, monospace`;
    ctx.textBaseline = 'alphabetic';
    for (let i = 0; i <= 3; i++) {
      const t = t0 + (span * i) / 3;
      ctx.textAlign = i === 0 ? 'left' : i === 3 ? 'right' : 'center';
      ctx.fillText(`${t.toFixed(0)} h`, pad.l + (pw * i) / 3, size.h - 3);
    }

    for (const c of channels) {
      ctx.strokeStyle = c.color;
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      data.forEach((p, i) => {
        const x = xOf(p.t);
        const y = yOf(p[c.key] ?? 0, c);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    if (hover !== null && data[hover]) {
      const hp = data[hover];
      const x = xOf(hp.t);
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const c of channels) {
        ctx.fillStyle = c.color;
        ctx.beginPath();
        ctx.arc(x, yOf(hp[c.key] ?? 0, c), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [data, size, hover, channels]);

  const onMove = (e: React.PointerEvent) => {
    if (data.length < 2) return;
    const r = e.currentTarget.getBoundingClientRect();
    const f = clamp((e.clientX - r.left - 4) / Math.max(r.width - 8, 1), 0, 1);
    const t = data[0].t + f * (data[data.length - 1].t - data[0].t);
    let best = 0;
    for (let i = 1; i < data.length; i++) if (Math.abs(data[i].t - t) < Math.abs(data[best].t - t)) best = i;
    setHover(best);
  };

  const shown = hover !== null && data[hover] ? data[hover] : data[data.length - 1];

  return (
    <div className="flex h-full min-h-37.5 min-w-0 flex-col gap-1 rounded-xl border border-slate-800 bg-slate-950/70 p-2.5">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 font-mono text-[10px]">
        <span className="flex items-center gap-1.5 text-slate-400"><Activity className="h-3 w-3 text-cyan-400" />{title}</span>
        <div className="flex flex-wrap gap-x-3 gap-y-0.5">
          {channels.map((c) => (
            <span key={c.key} style={{ color: c.color }} className="whitespace-nowrap">
              ● {c.label} {shown ? (shown[c.key] ?? 0).toFixed(c.digits ?? 1) : '–'} <span className="text-slate-500">{c.unit}</span>
            </span>
          ))}
        </div>
      </div>
      <div ref={wrapRef} className="relative min-h-22.5 flex-1">
        <canvas ref={canvasRef} onPointerMove={onMove} onPointerLeave={() => setHover(null)} className="absolute inset-0 h-full w-full touch-none" />
      </div>
    </div>
  );
});

/* ============================================================================
 * 10. THREE.JS PROCESS-TRAIN SCENE
 *     - built once, driven by a live-data ref (no rebuilds on state change)
 *     - sized by ResizeObserver on its own container, so it scales with the grid
 *     - camera auto-fits the focused stage or the whole train for any aspect
 * ========================================================================== */
const STAGE_SPACING = 4.6;
const stageX = (i: number) => (i - 3) * STAGE_SPACING;
const TRAIN_HALF_WIDTH = 3 * STAGE_SPACING + 2.4;

interface SceneLive {
  focus: number; //            -1 = whole train
  nonce: number; //            bump to re-fit the camera
  visible: boolean; //         false while another tab is showing: skip rendering
  active: number;
  doneCount: number;
  running: boolean;
  rpm: number;
  sparge: number;
  temperature: number;
  shear: number;
  dissolvedOxygen: number;
  upSeverity: 0 | 1 | 2;
  progress: number;
  pressureNorm: number;
  loadFrac: number;
  breakthrough: number;
  ph: number;
  volumeFrac: number;
  label: string;
}

interface HotspotInfo { name: string; desc: string; stage: number }
interface HotspotDef extends HotspotInfo { pos: [number, number, number] }

const HOTSPOTS: HotspotDef[] = [
  { stage: 0, name: 'Dissolved-oxygen probe', pos: [-0.95, 0.2, 0.35], desc: 'Optical pO₂ probe. Oxygen supply (kLa) must keep up with cell uptake or growth and titer collapse.' },
  { stage: 0, name: 'pH probe', pos: [0.95, 0.1, -0.35], desc: 'Solid-state pH sensor. The controller doses base to hold the setpoint against lactate acidification.' },
  { stage: 0, name: 'Dual Rushton turbines', pos: [0, 0.7, 0], desc: 'Radial-flow impellers for gas dispersion. Tip speed drives both mixing and cell-damaging shear.' },
  { stage: 0, name: 'Glucose feed line', pos: [-0.8, 1.2, 0.5], desc: 'Peristaltic line metering concentrated glucose. Too little starves cells; too much makes lactate.' },
  { stage: 1, name: 'Depth filter housing', pos: [0.3, 0.9, 0.55], desc: 'Porous depth media traps cells and debris. Pressure climbs as the filter loads; a blocked filter ends the batch.' },
  { stage: 2, name: 'Capture column', pos: [0, 1.4, 0.45], desc: 'Affinity/ion-exchange resin. Load past the dynamic binding capacity and product breaks through to waste.' },
  { stage: 3, name: 'Low-pH hold tank', pos: [0, 1.05, 0.9], desc: 'Acidic hold inactivates enveloped viruses. The clearance (LRV) must reach 4 log or the batch is rejected.' },
  { stage: 4, name: 'Polishing columns', pos: [0, 1.0, 0.5], desc: 'Flow-through resin binds host-cell protein, DNA and aggregates while antigen passes through.' },
  { stage: 5, name: 'TFF cassette holder', pos: [0.1, 0.8, 0.55], desc: 'Tangential-flow membranes concentrate the antigen and exchange it into formulation buffer.' },
  { stage: 6, name: 'Filling needle', pos: [-0.12, 0.35, 0.4], desc: 'Fill speed trades throughput against dose accuracy. Rejects rise sharply past the line\'s validated speed.' },
];

const C = { cyan: 0x00f0ff, amber: 0xf59e0b, red: 0xef4444, green: 0x10b981, violet: 0xa78bfa, rose: 0xf43f5e };

function ProcessTrainScene({ live, onSelect }: { live: SceneLive; onSelect: (h: HotspotInfo) => void }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef(live);
  liveRef.current = live;
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#030712');
    scene.fog = new THREE.FogExp2('#030712', 0.012);

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    const renderer = (() => {
      try {
        return new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
      } catch {
        return null;
      }
    })();
    if (!renderer) {
      setFailed(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const canvas = renderer.domElement;
    canvas.className = 'absolute inset-0 block h-full w-full touch-none';
    mount.appendChild(canvas);

    const grid = new THREE.GridHelper(80, 160, 0x06b6d4, 0x1e293b);
    grid.position.y = -1.25;
    scene.add(grid);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2.5;
    controls.maxDistance = 45;
    controls.maxPolarAngle = Math.PI / 2 + 0.15;

    /* ---- per-stage holographic materials ---- */
    const stageMats = STAGES.map(() => ({
      line: new THREE.LineBasicMaterial({ color: C.cyan, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending }),
      fill: new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.08, depthWrite: false, blending: THREE.AdditiveBlending }),
    }));
    const groups = STAGES.map((_, i) => {
      const g = new THREE.Group();
      g.position.x = stageX(i);
      scene.add(g);
      return g;
    });

    const holoBox = (i: number, w: number, h: number, d: number, x: number, y: number, z: number) => {
      const g = new THREE.Group();
      const geo = new THREE.BoxGeometry(w, h, d);
      g.add(new THREE.Mesh(geo, stageMats[i].fill));
      g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), stageMats[i].line));
      g.position.set(x, y, z);
      groups[i].add(g);
      return g;
    };
    const holoCyl = (i: number, rt: number, rb: number, h: number, seg: number, x: number, y: number, z: number) => {
      const g = new THREE.Group();
      const geo = new THREE.CylinderGeometry(rt, rb, h, seg);
      g.add(new THREE.Mesh(geo, stageMats[i].fill));
      g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 20), stageMats[i].line));
      g.position.set(x, y, z);
      groups[i].add(g);
      return g;
    };
    const fluid = (i: number, r: number, h: number, x: number, y: number, z: number, opacity = 0.2) => {
      const mat = new THREE.MeshBasicMaterial({ color: C.cyan, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false });
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 24), mat);
      mesh.position.set(x, y, z);
      groups[i].add(mesh);
      return { mat, mesh, baseY: y, baseH: h };
    };
    const pipe = (pts: [number, number, number][]) => {
      const mat = new THREE.LineBasicMaterial({ color: C.cyan, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending });
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p))), mat));
      return mat;
    };
    const labelJobs: (() => void)[] = [];
    const makeLabel = (text: string) => {
      const c = document.createElement('canvas');
      c.width = 512;
      c.height = 96;
      const tex = new THREE.CanvasTexture(c);
      const paint = () => {
        const g = c.getContext('2d')!;
        g.clearRect(0, 0, 512, 96);
        g.font = `600 38px ${fontData.style.fontFamily}, monospace`;
        g.fillStyle = '#67e8f9';
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText(text, 256, 48);
        tex.needsUpdate = true;
      };
      paint();
      labelJobs.push(paint);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
      sp.scale.set(2.8, 0.52, 1);
      return sp;
    };
    let disposed = false;
    // repaint labels once the web font has really loaded
    void document.fonts?.ready.then(() => { if (!disposed) labelJobs.forEach((j) => j()); });
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    const tmpColor = new THREE.Color();
    const heat = (mat: THREE.MeshBasicMaterial, n: number) => {
      const v = clamp(n, 0, 1.2);
      if (v < 0.6) mat.color.setHex(C.cyan).lerp(tmpColor.setHex(C.amber), v / 0.6);
      else mat.color.setHex(C.amber).lerp(tmpColor.setHex(C.red), (v - 0.6) / 0.6);
    };

    /* ---- decks, labels ---- */
    STAGES.forEach((st, i) => {
      holoBox(i, 3.6, 0.12, 2.6, 0, -0.9, 0);
      const label = makeLabel(st.short.toUpperCase());
      label.position.set(0, 3.05, 0);
      groups[i].add(label);
    });

    /* ---- stage 0: bioreactor ---- */
    [[-1.5, -1.0], [1.5, -1.0], [-1.5, 1.0], [1.5, 1.0]].forEach(([x, z]) => holoBox(0, 0.06, 2.7, 0.06, x, 0.45, z));
    holoBox(0, 3.06, 0.06, 2.06, 0, 1.8, 0);
    holoCyl(0, 1.05, 1.05, 2.3, 24, 0, 0.45, 0);
    holoCyl(0, 0.98, 0.98, 2.2, 24, 0, 0.45, 0);
    const brothFluid = fluid(0, 0.92, 1.85, 0, 0.35, 0);
    holoCyl(0, 0.3, 0.3, 0.6, 16, 0, 1.9, 0);
    holoBox(0, 0.7, 0.45, 0.04, 1.9, 1.3, 0);
    const impeller = new THREE.Group();
    impeller.position.set(0, 0.45, 0);
    const shaft = new THREE.Group();
    {
      const geo = new THREE.CylinderGeometry(0.04, 0.04, 2.2, 12);
      shaft.add(new THREE.Mesh(geo, stageMats[0].fill));
      shaft.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 20), stageMats[0].line));
    }
    impeller.add(shaft);
    [-0.6, 0.25].forEach((y) => {
      const ring = new THREE.Group();
      ring.position.y = y;
      for (let k = 0; k < 6; k++) {
        const th = (k * Math.PI) / 3;
        const geo = new THREE.BoxGeometry(0.24, 0.12, 0.02);
        const b = new THREE.Group();
        b.add(new THREE.Mesh(geo, stageMats[0].fill));
        b.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), stageMats[0].line));
        b.position.set(0.2 * Math.cos(th), 0, 0.2 * Math.sin(th));
        b.rotation.y = -th; // radial blades
        ring.add(b);
      }
      impeller.add(ring);
    });
    groups[0].add(impeller);

    const BUBBLES = 100;
    const bPos = new Float32Array(BUBBLES * 3);
    const bVel = new Float32Array(BUBBLES);
    for (let p = 0; p < BUBBLES; p++) {
      const a = Math.random() * Math.PI * 2;
      const r = 0.05 + Math.random() * 0.4;
      bPos[p * 3] = Math.cos(a) * r;
      bPos[p * 3 + 1] = -0.55 + Math.random() * 1.8;
      bPos[p * 3 + 2] = Math.sin(a) * r;
      bVel[p] = 0.5 + Math.random() * 0.9; // units/s
    }
    const bGeo = new THREE.BufferGeometry();
    bGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
    groups[0].add(new THREE.Points(bGeo, new THREE.PointsMaterial({ color: C.cyan, size: 0.05, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending })));

    /* ---- stage 1: clarification ---- */
    holoCyl(1, 0.5, 0.5, 1.0, 16, -1.2, -0.3, 0.6);
    holoCyl(1, 0.58, 0.58, 1.9, 20, 0.3, 0.05, 0);
    holoCyl(1, 0.3, 0.58, 0.3, 20, 0.3, 1.15, 0);
    holoBox(1, 0.5, 0.4, 0.5, -1.2, -0.65, -0.6);
    const clarFluid = fluid(1, 0.5, 1.55, 0.3, 0.05, 0);

    /* ---- stage 2: capture column ---- */
    holoCyl(2, 0.42, 0.42, 2.6, 20, 0, 0.4, 0);
    holoCyl(2, 0.35, 0.35, 0.9, 16, -1.3, -0.4, 0.4);
    holoCyl(2, 0.35, 0.35, 0.9, 16, 1.3, -0.4, 0.4);
    const bed = fluid(2, 0.36, 1, 0, -0.5, 0, 0.35);

    /* ---- stage 3: viral inactivation ---- */
    holoCyl(3, 0.9, 0.9, 1.9, 24, 0, 0.1, 0);
    holoCyl(3, 0.03, 0.03, 2.2, 8, 0, 0.35, 0);
    holoBox(3, 0.5, 0.05, 0.05, 0, -0.55, 0);
    const holdFluid = fluid(3, 0.84, 1.6, 0, 0.0, 0);

    /* ---- stage 4: polishing ---- */
    holoCyl(4, 0.4, 0.4, 1.8, 20, -0.7, 0, 0);
    holoCyl(4, 0.4, 0.4, 1.8, 20, 0.7, 0, 0);
    const polA = fluid(4, 0.34, 1.4, -0.7, 0, 0);
    const polB = fluid(4, 0.34, 1.4, 0.7, 0, 0);
    holoBox(4, 0.4, 0.5, 0.4, 0, -0.6, 0.9);

    /* ---- stage 5: UF/DF ---- */
    for (let k = 0; k < 8; k++) holoBox(5, 0.05, 1.3, 0.9, 0.1 + k * 0.14, 0.0, 0);
    holoBox(5, 0.08, 1.5, 1.0, 0.0, 0.0, 0);
    holoBox(5, 0.08, 1.5, 1.0, 1.2, 0.0, 0);
    holoCyl(5, 0.55, 0.55, 1.3, 20, -1.2, -0.15, 0);
    holoBox(5, 0.4, 0.4, 0.4, 0.6, -0.65, 0.9);
    const retFluid = fluid(5, 0.5, 1.15, -1.2, -0.15, 0);

    /* ---- stage 6: sterile filter + fill-finish ---- */
    holoCyl(6, 0.22, 0.22, 0.8, 16, -1.3, 0.0, 0);
    holoBox(6, 2.8, 0.08, 0.6, 0.2, -0.65, 0);
    holoBox(6, 0.5, 0.3, 0.5, -0.12, 0.45, 0);
    holoCyl(6, 0.02, 0.02, 0.3, 6, -0.12, 0.2, 0);
    const VIALS = 12;
    const vialGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.28, 10);
    const vialEdges = new THREE.EdgesGeometry(vialGeo);
    const liquidGeo = new THREE.CylinderGeometry(0.055, 0.055, 0.2, 10);
    const vials = Array.from({ length: VIALS }, () => {
      const g = new THREE.Group();
      g.add(new THREE.Mesh(vialGeo, stageMats[6].fill));
      g.add(new THREE.LineSegments(vialEdges, stageMats[6].line));
      const liq = new THREE.Mesh(liquidGeo, new THREE.MeshBasicMaterial({ color: C.green, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false }));
      liq.position.y = -0.03;
      g.add(liq);
      g.position.y = -0.47;
      groups[6].add(g);
      return { g, liq };
    });
    let vialPhase = 0;

    /* ---- pipes between stages ---- */
    const pipes: THREE.LineBasicMaterial[] = [];
    for (let i = 0; i < STAGES.length - 1; i++) {
      pipes.push(pipe([[stageX(i) + 1.8, -0.5, 0.9], [stageX(i) + 2.3, -0.5, 0.9], [stageX(i + 1) - 2.3, -0.5, 0.9], [stageX(i + 1) - 1.8, -0.5, 0.9]]));
    }

    /* ---- clickable sensor nodes ---- */
    const hitMeshes: THREE.Mesh[] = [];
    const pulses: THREE.Mesh[] = [];
    const nodeGeo = new THREE.SphereGeometry(0.07, 12, 12);
    const hitGeo = new THREE.SphereGeometry(0.26, 8, 8);
    const nodeMat = new THREE.MeshBasicMaterial({ color: C.cyan, wireframe: true });
    const hitMat = new THREE.MeshBasicMaterial({ visible: false });
    HOTSPOTS.forEach((h) => {
      const vis = new THREE.Mesh(nodeGeo, nodeMat);
      vis.position.set(...h.pos);
      groups[h.stage].add(vis);
      pulses.push(vis);
      const hit = new THREE.Mesh(hitGeo, hitMat);
      hit.position.set(...h.pos);
      hit.userData = { info: { name: h.name, desc: h.desc, stage: h.stage } as HotspotInfo };
      groups[h.stage].add(hit);
      hitMeshes.push(hit);
    });

    /* ---- camera framing ---- */
    const targetFor = (focus: number) => new THREE.Vector3(focus === -1 ? 0 : stageX(focus), 0.5, 0);
    const distanceFor = (focus: number) => {
      const t = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      const halfW = focus === -1 ? TRAIN_HALF_WIDTH : 3.0;
      const distH = 2.6 / t;
      const distW = halfW / (t * Math.max(camera.aspect, 0.2));
      return Math.max(distH, distW) * 1.08;
    };
    const dir0 = new THREE.Vector3(0.0, 0.3, 1).normalize();
    let anim = 120;
    let lastFocus = liveRef.current.focus;
    let lastNonce = liveRef.current.nonce;
    {
      const d = distanceFor(lastFocus);
      controls.target.copy(targetFor(lastFocus));
      camera.position.copy(controls.target).addScaledVector(dir0, d);
    }

    /* ---- sizing: follow the container, not the window ---- */
    const resize = () => {
      const w = Math.max(1, mount.clientWidth);
      const h = Math.max(1, mount.clientHeight);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      anim = Math.max(anim, 40);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    /* ---- picking (click without drag) ---- */
    const raycaster = new THREE.Raycaster();
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const onUp = (e: PointerEvent) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { down = null; return; }
      down = null;
      const r = canvas.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
      const hit = raycaster.intersectObjects(hitMeshes, false)[0];
      if (hit) {
        selectRef.current(hit.object.userData.info as HotspotInfo);
        audio.chime(1040, 0.1);
      }
    };
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    /* ---- render loop ---- */
    const clock = new THREE.Clock();
    let raf = 0;
    let time = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      time += dt;
      const d = liveRef.current;
      if (!d.visible) return;

      // stage emphasis: active bright, finished green, pending dim
      stageMats.forEach((m, i) => {
        const done = i < d.doneCount;
        const want = d.focus === -1 ? (i === d.active ? 1 : done ? 0.7 : 0.5) : i === d.focus ? 1 : done ? 0.55 : 0.3;
        m.line.opacity += (want * 0.85 - m.line.opacity) * 0.1;
        m.fill.opacity += (want * 0.09 - m.fill.opacity) * 0.1;
        m.line.color.setHex(done ? C.green : C.cyan);
        m.fill.color.setHex(done ? 0x059669 : 0x06b6d4);
      });
      pipes.forEach((p, i) => {
        p.color.setHex(i < d.active - 1 ? C.green : i === d.active - 1 ? C.cyan : 0x1e5a6b);
        p.opacity = i < d.doneCount ? 0.8 : 0.4;
      });

      // stage 0
      if (d.running && d.active === 0) impeller.rotation.y += Math.min((d.rpm / 800) * 8, 8) * dt;
      if (d.upSeverity === 2) brothFluid.mat.color.setHex(C.red);
      else if (d.upSeverity === 1) brothFluid.mat.color.setHex(C.amber);
      else brothFluid.mat.color.setHex(C.cyan);
      const bSpeed = d.running && d.active === 0 ? 1 + d.sparge * 2.5 : 0.1;
      for (let b = 0; b < BUBBLES; b++) {
        bPos[b * 3 + 1] += bVel[b] * bSpeed * dt;
        if (bPos[b * 3 + 1] > 1.25) bPos[b * 3 + 1] = -0.55;
      }
      bGeo.attributes.position.needsUpdate = true;

      // downstream stages: only the active one reads live data
      heat(clarFluid.mat, d.active === 1 ? d.pressureNorm : 0);
      if (d.active === 2) {
        const f = clamp(0.04 + d.loadFrac * 0.96, 0.04, 1) * 1.8;
        bed.mesh.scale.y = f;
        bed.mesh.position.y = -0.7 + (bed.baseH * f) / 2;
        if (d.label === 'ELUTE') bed.mat.color.setHex(C.violet);
        else heat(bed.mat, d.breakthrough / 30);
      } else {
        bed.mesh.scale.y = 1.2;
        bed.mesh.position.y = -0.7 + 0.6;
        bed.mat.color.setHex(C.cyan);
      }
      if (d.active === 3) holdFluid.mat.color.setHex(C.cyan).lerp(tmpColor.setHex(C.rose), clamp((7 - d.ph) / 3.7, 0, 1));
      else holdFluid.mat.color.setHex(C.cyan);
      const pol = d.active === 4 ? d.progress : 0;
      polA.mat.color.setHex(C.cyan).lerp(tmpColor.setHex(C.green), pol);
      polB.mat.color.setHex(C.cyan).lerp(tmpColor.setHex(C.green), clamp(pol * 1.3 - 0.2, 0, 1));
      const vf = d.active === 5 ? clamp(d.volumeFrac, 0.12, 1) : 1;
      retFluid.mesh.scale.y = vf;
      retFluid.mesh.position.y = retFluid.baseY - (retFluid.baseH * (1 - vf)) / 2;

      if (d.active === 6 && d.running) vialPhase += dt * 0.12;
      vials.forEach((v, k) => {
        const u = (k / VIALS + vialPhase) % 1;
        v.g.position.x = -1.5 + u * 3.4;
        v.liq.scale.y = u > 0.45 ? clamp((u - 0.45) / 0.08, 0.001, 1) : 0.001;
      });

      if (!reduceMotion) pulses.forEach((m, i) => m.scale.setScalar(1 + 0.25 * Math.sin(time * 3 + i)));

      // camera: ease toward the requested framing, then hand control back to the user
      if (d.focus !== lastFocus || d.nonce !== lastNonce) {
        lastFocus = d.focus;
        lastNonce = d.nonce;
        anim = 90;
      }
      if (anim > 0) {
        anim--;
        const want = targetFor(d.focus);
        const delta = want.clone().sub(controls.target).multiplyScalar(0.12);
        controls.target.add(delta);
        camera.position.add(delta);
        const off = camera.position.clone().sub(controls.target);
        const cur = off.length();
        const wantD = distanceFor(d.focus);
        off.setLength(cur + (wantD - cur) * 0.12);
        camera.position.copy(controls.target).add(off);
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointerup', onUp);
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
        else if (mat) {
          (mat as THREE.SpriteMaterial).map?.dispose?.();
          mat.dispose();
        }
      });
      if (mount.contains(canvas)) mount.removeChild(canvas);
      renderer.dispose();
      renderer.forceContextLoss();
    };
  }, []);

  if (failed) {
    return (
      <div className="absolute inset-0 flex items-center justify-center p-6 text-center font-mono text-xs text-slate-400">
        WebGL is not available in this browser. The simulation, charts and controls still work.
      </div>
    );
  }
  return <div ref={mountRef} className="absolute inset-0 cursor-grab overflow-hidden active:cursor-grabbing" />;
}

/* ============================================================================
 * 11. SMALL PRESENTATIONAL PIECES
 * ========================================================================== */
const TONE = {
  ok: 'text-cyan-300',
  good: 'text-emerald-300',
  warn: 'text-amber-300',
  bad: 'text-rose-300',
  plain: 'text-white',
} as const;
type Tone = keyof typeof TONE;

const Metric = ({ label, value, unit, tone = 'plain' }: { label: string; value: string; unit?: string; tone?: Tone }) => (
  <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-950/70 px-2.5 py-1.5">
    <div className="truncate font-mono text-[10px] text-slate-500">{label}</div>
    <div className={`truncate font-mono text-sm font-bold ${TONE[tone]}`}>
      {value}
      {unit && <span className="ml-1 text-[10px] font-normal text-slate-500">{unit}</span>}
    </div>
  </div>
);

function buildMetrics(snap: Snapshot): { label: string; value: string; unit?: string; tone?: Tone }[] {
  if (snap.phase === 'UPSTREAM') {
    const u = snap.up;
    const viab = (100 * u.biomass) / (u.biomass + u.debris);
    return [
      { label: 'Runtime', value: snap.upHours.toFixed(1), unit: `h · day ${(snap.upHours / 24).toFixed(1)}`, tone: 'ok' },
      { label: 'Viable cell density', value: (u.biomass * BIO.cellsPerGram).toFixed(1), unit: '×10⁶/mL', tone: 'good' },
      { label: 'Biomass', value: u.biomass.toFixed(2), unit: 'g/L DCW' },
      { label: 'Viability', value: viab.toFixed(1), unit: '%', tone: viab < 80 ? 'warn' : 'plain' },
      { label: 'Antigen titer', value: u.titer.toFixed(2), unit: 'g/L', tone: 'ok' },
      { label: 'Glucose', value: u.substrate.toFixed(1), unit: 'g/L', tone: u.substrate < LIMITS.glucoseLow ? 'warn' : 'plain' },
      { label: 'Lactate', value: u.lactate.toFixed(2), unit: 'g/L' },
      { label: 'dO₂', value: u.dissolvedOxygen.toFixed(0), unit: '%', tone: u.dissolvedOxygen < LIMITS.doWarn ? 'warn' : 'plain' },
      { label: 'Broth pH', value: u.brothPh.toFixed(2) },
      { label: 'Monomer purity', value: u.monomerPurity.toFixed(1), unit: '%', tone: u.monomerPurity >= SPEC.minPurity ? 'ok' : 'bad' },
    ];
  }
  const d = snap.ds;
  if (!d) return [];
  const key = DS_ORDER[d.stage];
  const list: { label: string; value: string; unit?: string; tone?: Tone }[] = [
    { label: 'Stage', value: STAGES[d.stage + 1].short, unit: d.live.label.toLowerCase(), tone: 'ok' },
    { label: 'Stage progress', value: (d.progress * 100).toFixed(0), unit: '%' },
    { label: 'Recovery', value: ((d.product / d.startProduct) * 100).toFixed(1), unit: '%', tone: 'good' },
    { label: 'Antigen', value: (d.product / 1000).toFixed(2), unit: 'kg' },
    { label: 'Volume', value: d.volume.toFixed(0), unit: 'L' },
    { label: 'Purity', value: d.purity.toFixed(2), unit: '%', tone: d.purity >= SPEC.minPurity ? 'ok' : 'bad' },
    { label: 'HCP', value: d.hcp < 1000 ? d.hcp.toFixed(1) : d.hcp.toExponential(1), unit: 'ppm', tone: d.hcp <= SPEC.maxHcpPpm ? 'good' : 'plain' },
    { label: 'Residual DNA', value: d.dna < 1000 ? d.dna.toFixed(2) : d.dna.toExponential(1), unit: 'ng/mg' },
  ];
  if (d.live.limit > 0) list.push({ label: key === 'UFDF' ? 'TMP' : 'Pressure', value: d.live.pressure.toFixed(2), unit: `/ ${d.live.limit.toFixed(1)} bar`, tone: d.live.pressure > 0.8 * d.live.limit ? 'warn' : 'plain' });
  if (key === 'CLARIFY') list.push({ label: 'Turbidity', value: d.live.turbidity.toFixed(1), unit: 'NTU' });
  if (key === 'CAPTURE') list.push({ label: 'Breakthrough', value: d.live.breakthrough.toFixed(1), unit: '%', tone: d.live.breakthrough > 10 ? 'warn' : 'plain' });
  if (key === 'VIRAL') list.push({ label: 'Viral LRV', value: d.lrv.toFixed(1), unit: `/ ${SPEC.minLrv.toFixed(1)}`, tone: d.lrv >= SPEC.minLrv ? 'good' : 'warn' });
  if (key === 'FILL') list.push({ label: 'Vials filled', value: (d.vials.filled / 1e6).toFixed(2), unit: `/ ${(d.vials.total / 1e6).toFixed(2)} M` });
  return list;
}

function ReleasePanel({ ds, dsIn }: { ds: DsState; dsIn: DsInputs }) {
  const checks = releaseChecks(ds, dsIn);
  const done = ds.status === 'COMPLETE';
  const ok = done && checks.every((c) => c.pass);
  const doses = (ds.product * 1e6) / dsIn.doseUg;
  const reject = ds.vials.total > 0 ? ds.vials.rejected / ds.vials.filled : 0;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-mono text-xs font-bold text-white">Batch release</h3>
        <span className={`rounded-full border px-2 py-0.5 font-mono text-[10px] font-bold ${ds.status === 'FAILED' ? 'border-rose-700 bg-rose-950 text-rose-300' : done ? (ok ? 'border-emerald-700 bg-emerald-950 text-emerald-300' : 'border-rose-700 bg-rose-950 text-rose-300') : 'border-slate-700 bg-slate-900 text-slate-400'}`}>
          {ds.status === 'FAILED' ? 'REJECTED' : done ? (ok ? 'CONFORMS' : 'OUT OF SPEC') : 'IN PROCESS'}
        </span>
      </div>
      <div className="grid gap-2 grid-cols-[repeat(auto-fit,minmax(220px,1fr))]">
        <ul className="space-y-1">
          {checks.map((c) => (
            <li key={c.name} className="flex items-center justify-between gap-2 font-mono text-[11px]">
              <span className="flex min-w-0 items-center gap-1.5 text-slate-300">
                {c.pass ? <Check className="h-3 w-3 shrink-0 text-emerald-400" /> : <X className="h-3 w-3 shrink-0 text-rose-400" />}
                <span className="truncate">{c.name}</span>
              </span>
              <span className={c.pass ? 'text-emerald-300' : 'text-rose-300'}>{c.value} <span className="text-slate-500">({c.spec})</span></span>
            </li>
          ))}
        </ul>
        <div>
          <div className="mb-1 font-mono text-[10px] text-slate-500">Cumulative recovery by unit operation</div>
          <div className="space-y-1">
            {ds.recovery.map((r, i) => (
              <div key={i} className="flex items-center gap-2 font-mono text-[10px] text-slate-400">
                <span className="w-12 shrink-0">{STAGES[i + 1].short}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded bg-slate-800"><div className="h-full bg-cyan-400" style={{ width: `${clamp(r, 0, 100)}%` }} /></div>
                <span className="w-10 shrink-0 text-right">{r.toFixed(0)}%</span>
              </div>
            ))}
            {ds.recovery.length === 0 && <div className="font-mono text-[10px] text-slate-600">No unit operation finished yet.</div>}
          </div>
          {done && <div className="mt-2 font-mono text-[11px] text-slate-300">≈ {(doses * (1 - reject) / 1e6).toFixed(0)} M doses · reject rate {(reject * 100).toFixed(1)} %</div>}
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
 * 12. STATIC CONTENT (data-driven so the tabs stay compact)
 * ========================================================================== */
const SCIENCE_CARDS: { id: string; title: string; sym: string; tone: string; lines: string[]; body: string }[] = [
  { id: '01', title: 'Cell kinetics', sym: 'dX/dt', tone: 'text-cyan-400', lines: ['μ = μmax · S/(Ks+S) · f(T) · f(pH) · DO/(DO+Ko₂) · 1/(1+(La/Li)²)', 'dX/dt = (μ − kd − k_lysis) · X'], body: 'Monod growth tempered by quadratic temperature and pH penalties, oxygen limitation and lactate inhibition. Dead and lysed cells accumulate as debris, which later loads the depth filter.' },
  { id: '02', title: 'Hydrodynamic shear', sym: 'τ (Pa)', tone: 'text-rose-400', lines: ['τ = k · rpm^1.96', 'k_lysis = 0.22 · max(0, τ − 1.5)'], body: 'Empirical correlation. Above the critical shear the membrane ruptures and product aggregates. Real shear scales with impeller diameter and power per volume, so recalibrate for your vessel.' },
  { id: '03', title: 'Oxygen transfer', sym: 'dDO/dt', tone: 'text-emerald-400', lines: ['kLa = α · rpm^1.18 + β · vvm', 'dDO/dt = kLa · (100 − DO) − qO₂ · X'], body: 'Supply balanced against uptake in %air-saturation per hour. qO₂ is calibrated so DO settles near 40 % at 8 g/L DCW.' },
  { id: '04', title: 'Capture breakthrough', sym: 'C/C₀', tone: 'text-amber-400', lines: ['DBC₁₀ = 60 · (1 − e^(−RT/1.5))', 'BT = 1/(1 + e^(−10(r − 1.22))),  r = load/DBC₁₀', 'yield = (1 − mean BT) · elution recovery(pH)'], body: 'Slower flow raises binding capacity. Overloading lets antigen leak through; lower elution pH recovers more but aggregates antigen and co-elutes host-cell protein.' },
  { id: '05', title: 'Viral inactivation', sym: 'LRV', tone: 'text-rose-400', lines: ['k = k₀ · 10^((4 − pH)·1.6) · 2^((T − 20)/10)', 'LRV = min(6.5, k · t_hold)  (spec ≥ 4)'], body: 'First-order inactivation. The step passes or fails the whole batch, so it is a hard gate, not a soft penalty.' },
  { id: '06', title: 'Depth filtration', sym: 'ΔP', tone: 'text-cyan-400', lines: ['capacity = 240/(1 + 0.3·debris)  (L/m²)', 'ΔP = 0.15 + 0.25·(flux/100) + 1.8·(load/capacity)²'], body: 'Fouling is quadratic in loading. Cell debris from the upstream run reduces capacity, so a stressed culture makes harvest harder.' },
  { id: '07', title: 'Tangential-flow UF/DF', sym: 'J', tone: 'text-emerald-400', lines: ['J = k·√(crossflow/5) · ln(Cg/Cb) · (1 − e^(−TMP/0.8))', 'DV = ∫ J·A dt / V_retentate'], body: 'Gel-polarisation flux. Concentrate to target, then diafilter for N diavolumes. High TMP × crossflow shears and aggregates the antigen.' },
  { id: '08', title: 'Polishing & fill', sym: 'LRV_HCP', tone: 'text-amber-400', lines: ['LRV = 3·f(conductivity)·f(load)', 'rejects = 0.5 % + 4 %·((speed − 420)/180)²'], body: 'Flow-through clearance falls as conductivity or load leaves the window. Fill speed past the validated rate drives vial rejects.' },
];

const CITATIONS: { text: string; verified: boolean; url?: string }[] = [
  { text: 'Africa CDC. (2022). Partnerships for African Vaccine Manufacturing (PAVM) framework for action: Version 1. African Union.', verified: true, url: 'https://africacdc.org/download/partnerships-for-african-vaccine-manufacturing-pavm-framework-for-action/' },
  { text: 'Gavi, the Vaccine Alliance. (n.d.). African Vaccine Manufacturing Accelerator (AVMA).', verified: true, url: 'https://www.gavi.org/avma' },
  { text: 'Gavi, the Vaccine Alliance. (n.d.). AVMA: Frequently asked questions.', verified: true, url: 'https://www.gavi.org/programmes-impact/types-support/regional-manufacturing-strategy/avma/faqs' },
  { text: 'Erondu, N. (2022, August 3). Now is the moment to launch an African vaccine industry. Chatham House.', verified: true, url: 'https://www.chathamhouse.org/publications/the-world-today/2022-08/now-moment-launch-african-vaccine-industry' },
  { text: 'Kenya BioVax Institute. (n.d.). Who we are.', verified: true, url: 'https://biovax.go.ke/who-we-are/' },
  { text: 'International Council for Harmonisation. (2009). Pharmaceutical development Q8(R2). ICH Harmonised Tripartite Guideline.', verified: true },
  { text: 'Rathore, A. S., & Winkle, H. (2009). Quality by design for biopharmaceuticals. Nature Biotechnology, 27(1), 26–34.', verified: true },
  { text: 'Wang, W. (2005). Protein aggregation and its inhibition in biopharmaceutics. International Journal of Pharmaceutics, 289(1–2), 1–30.', verified: true },
  { text: 'Narayanan, H., Dingfelder, F., Butté, A., Sokolov, M., & Arosio, P. (2020). A review of mathematical models for the simulation of bioprocesses. Processes, 8(9), 1145.', verified: false },
];


/* ============================================================================
 * 12b. SOVEREIGNTY CONTENT
 *      Sourced from Africa CDC's PAVM Framework for Action v1 (2022, read in full),
 *      Gavi's AVMA pages, and Kenya BioVax Institute's own pages (all read Oct 2026).
 *      The 2022 Framework figures are the plan's baseline, not a progress report.
 * ========================================================================== */
const SRC = {
  ffa: { short: 'Africa CDC FFA v1 (2022)', label: 'Africa CDC: PAVM Framework for Action, Version 1 (2022)', url: 'https://africacdc.org/download/partnerships-for-african-vaccine-manufacturing-pavm-framework-for-action/' },
  avma: { short: 'Gavi AVMA', label: 'Gavi: African Vaccine Manufacturing Accelerator (AVMA)', url: 'https://www.gavi.org/avma' },
  avmaFaq: { short: 'Gavi AVMA FAQ', label: 'Gavi: AVMA frequently asked questions', url: 'https://www.gavi.org/programmes-impact/types-support/regional-manufacturing-strategy/avma/faqs' },
  avmaLaunch: { short: 'Gavi launch release', label: 'Gavi: AVMA launch at the Global Forum for Vaccine Sovereignty and Innovation, Paris, 20 June 2024', url: 'https://gavi.org/investing-gavi/resource-mobilisation-process/protecting-our-future' },
  phahm: { short: 'World Health Summit', label: 'World Health Summit 2025 session: four years of support to vaccine production in Africa (PAVM renamed PHAHM)', url: 'https://www.worldhealthsummit.org/sessions/67b5a017-9d44-42e3-98fc-6c05cff585af' },
  chatham: { short: 'Chatham House', label: 'Chatham House: Now is the moment to launch an African vaccine industry (Erondu, 2022)', url: 'https://www.chathamhouse.org/publications/the-world-today/2022-08/now-moment-launch-african-vaccine-industry' },
  biovax: { short: 'Kenya BioVax', label: 'Kenya BioVax Institute: Who we are', url: 'https://biovax.go.ke/who-we-are/' },
  biovaxKorea: { short: 'Kenya BioVax (Korea mission)', label: 'Kenya BioVax: Mission to Korea strengthens path to vaccine self-reliance (Oct 2025)', url: 'https://biovax.go.ke/kenya-biovax-mission-to-korea-strengthens-path-to-vaccine-self-reliance/' },
  biovaxTraining: { short: 'Kenya BioVax (CSIR training)', label: 'Kenya BioVax: African Biomanufacturing Workforce Training Programme at CSIR, May 2025', url: 'https://biovax.go.ke/?p=6272' },
  fellowship: { short: 'Africa CDC/AVMI fellowship call', label: 'Africa CDC / AVMI 2026 Industrial Fellowship Programme in Vaccine Manufacturing (call, via Opportunities for Africans)', url: 'https://www.opportunitiesforafricans.com/africa-cdc-avmi-2026-industrial-fellowship-programme-in-vaccine-manufacturing-fully-funded/' },
} as const;
type SrcKey = keyof typeof SRC;

const Cite = ({ s, where }: { s: SrcKey; where?: string }) => (
  <a href={SRC[s].url} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 font-mono text-[10px] text-cyan-400 underline decoration-cyan-900 underline-offset-2 hover:text-cyan-300">
    {SRC[s].short}{where ? `, ${where}` : ''}
    <ExternalLink className="h-2.5 w-2.5 shrink-0" aria-hidden />
  </a>
);

const SOV_STATS: { big: string; tone: string; label: string; note: string; src: SrcKey; where?: string }[] = [
  { big: '~1 %', tone: 'text-rose-400', label: 'of the vaccines used in Africa are made in Africa', note: 'Indian suppliers provided about 70 % of Gavi-procured supply at the time of the report.', src: 'ffa', where: 'Executive summary' },
  { big: '60 %', tone: 'text-cyan-400', label: 'target share of African vaccine doses made on the continent by 2040', note: 'Interim goals: 10 % by 2025 and 30 % by 2030.', src: 'ffa', where: 'Executive summary' },
  { big: '1.5 B', tone: 'text-emerald-400', label: 'doses a year of local capacity needed by 2040', note: 'Demand is forecast to grow from about 1 billion to over 2.7 billion doses a year.', src: 'ffa', where: 'Executive summary' },
  { big: '12,500', tone: 'text-amber-400', label: 'vaccine workforce (full-time staff) needed by 2040', note: 'Roughly 2,000 to 3,000 today. About 10,500 people need training.', src: 'ffa', where: 'Program 5' },
  { big: '$30 B', tone: 'text-violet-400', label: 'investment over 20 years to deliver the Framework', note: 'About $5 B one-off (plants, cold chain) and $25 B recurring.', src: 'ffa', where: 'Executive summary' },
  { big: '$1.2 B', tone: 'text-cyan-300', label: 'Gavi AVMA incentives (up to) over ten years from June 2024', note: 'Per-dose top-ups and milestone payments for vaccines made in Africa.', src: 'avma' },
];

const SOV_TIMELINE: { when: string; text: string; src: SrcKey; where?: string }[] = [
  { when: 'Apr 2021', text: 'African leaders meet at the AU and Africa CDC conference on expanding vaccine manufacturing and set the 60 % by 2040 ambition. PAVM is established under Africa CDC.', src: 'ffa', where: 'Introduction' },
  { when: '2022', text: 'Framework for Action v1 is published, with eight bold programmes.', src: 'ffa' },
  { when: 'Dec 2022', text: 'Gavi’s Board approves a four-pillar regional manufacturing strategy.', src: 'avmaFaq' },
  { when: 'Dec 2023', text: 'Gavi’s Board approves AVMA, one of those four pillars.', src: 'avmaFaq' },
  { when: 'Feb 2024', text: 'The African Union widens the mandate to all health products. PAVM is renamed the Platform for Harmonized African Health Products Manufacturing (PHAHM).', src: 'phahm' },
  { when: '20 Jun 2024', text: 'AVMA launches in Paris with twelve sovereign and philanthropic donors committing about US$1.2 billion.', src: 'avmaLaunch' },
  { when: '2040', text: 'Target year: 60 % of vaccine doses used in Africa are made in Africa.', src: 'ffa' },
];

const FFA_PROGRAMMES = [
  'African vaccines procurement pooling mechanism',
  'Vaccine manufacturing deal preparation facility',
  'Stronger regulators and regional centres of regulatory excellence',
  'Technology transfer and IP enablement unit',
  'Regional capability and capacity centres (talent)',
  'Vaccine R&D centres and a coordinating platform',
  'Advocacy for enabling trade policy',
  'Continental delivery and oversight',
];

const AVMA_TIERS = [
  { rate: '$0.50 / dose', what: 'Drug substance made in Africa, for AVMA Priority vaccines or platforms' },
  { rate: '$0.40 / dose', what: 'Drug substance made in Africa, for any other Gavi portfolio vaccine' },
  { rate: '$0.30 / dose', what: 'Only fill and finish done in Africa (capped at $1.00 per vial)' },
];

const SALUS_FIT = [
  { stage: 'Upstream bioreactor', skill: 'Oxygen transfer, shear, feeding strategy, reading a culture run' },
  { stage: 'Clarify, capture, polish', skill: 'Filter loading limits, column breakthrough, impurity clearance' },
  { stage: 'Viral inactivation', skill: 'Treating viral safety as a pass-or-fail gate' },
  { stage: 'UF/DF and fill', skill: 'Membrane trade-offs and fill-finish rejects' },
  { stage: 'Release', skill: 'Reading a certificate of analysis against specifications, and what a deviation means' },
];

function SovereigntyTab() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-10 p-6 md:p-10">
        <header className="space-y-3 border-b border-slate-800 pb-6">
          <span className="rounded-full border border-emerald-800 bg-emerald-950/60 px-3 py-1 font-mono text-xs text-emerald-400">Africa CDC · African Union · Gavi</span>
          <h1 className="pt-2 text-3xl font-extrabold text-white">Vaccine sovereignty and workforce capacity</h1>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-300">
            Africa set itself a goal in 2021: make 60 % of the vaccine doses it uses on the continent by 2040, up from about 1 %. Hitting it needs plants, regulators, financing and, above all, people who can run a bioprocess. This page lays out the plan as its authors wrote it, and where this simulator fits.
          </p>
          <p className="max-w-3xl rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 font-mono text-[11px] leading-relaxed text-slate-400">
            PAVM is now called PHAHM (see timeline). Figures marked FFA v1 are the 2022 baseline and targets, not current progress; check Africa CDC for newer updates before quoting them. Salus Tech is an independent educational project and is not endorsed by Africa CDC, the African Union or Gavi.
          </p>
        </header>

        <section aria-labelledby="sov-stats" className="space-y-3">
          <h2 id="sov-stats" className="text-lg font-bold text-white">The numbers behind the target</h2>
          <div className="grid gap-4 grid-cols-[repeat(auto-fit,minmax(240px,1fr))]">
            {SOV_STATS.map((st) => (
              <div key={st.label} className="flex flex-col gap-1.5 rounded-2xl border border-slate-800 bg-[#070d18] p-5">
                <div className={`font-mono text-3xl font-extrabold ${st.tone}`}>{st.big}</div>
                <div className="text-sm font-semibold leading-snug text-slate-100">{st.label}</div>
                <p className="text-xs leading-relaxed text-slate-400">{st.note}</p>
                <div className="mt-auto pt-1"><Cite s={st.src} where={st.where} /></div>
              </div>
            ))}
          </div>
        </section>

        <section aria-labelledby="sov-timeline" className="space-y-3">
          <h2 id="sov-timeline" className="text-lg font-bold text-white">From PAVM to PHAHM: how it got here</h2>
          <ol className="space-y-0 rounded-2xl border border-slate-800 bg-[#070d18] p-2">
            {SOV_TIMELINE.map((t) => (
              <li key={t.when} className="grid gap-x-4 gap-y-1 border-b border-slate-800/70 p-3 last:border-b-0 sm:grid-cols-[110px_1fr]">
                <div className="font-mono text-xs font-bold text-cyan-400">{t.when}</div>
                <div>
                  <p className="text-sm leading-relaxed text-slate-300">{t.text}</p>
                  <Cite s={t.src} where={t.where} />
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section aria-labelledby="sov-programmes" className="space-y-3">
          <h2 id="sov-programmes" className="text-lg font-bold text-white">The Framework’s eight bold programmes</h2>
          <div className="grid gap-3 grid-cols-[repeat(auto-fit,minmax(220px,1fr))]">
            {FFA_PROGRAMMES.map((name, i) => (
              <div key={name} className={`rounded-xl border p-3.5 ${i === 4 ? 'border-cyan-600/70 bg-cyan-950/30' : 'border-slate-800 bg-[#070d18]'}`}>
                <div className="font-mono text-[11px] text-slate-500">Programme {i + 1}</div>
                <div className="mt-0.5 text-sm leading-snug text-slate-100">{name}</div>
                {i === 4 && <div className="mt-1.5 text-[11px] text-cyan-300">The one this simulator speaks to</div>}
              </div>
            ))}
          </div>
          <Cite s="ffa" where="Executive summary" />
        </section>

        <section aria-labelledby="sov-ds" className="space-y-4 rounded-2xl border border-slate-800 bg-[#070d18] p-6">
          <h2 id="sov-ds" className="text-lg font-bold text-white">Why drug substance, not just fill and finish</h2>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-300">
            The Framework argues for an integrated industry: research, drug substance (the bioreactor and purification steps) and fill and finish. It estimates 23 plants are needed, 12 of them integrated drug-substance-and-fill plants and 11 fill-only. At the time of the report only five local companies did any drug substance work, and mostly at very small scale. <Cite s="ffa" where="§IV and §II" />
          </p>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-300">
            Gavi’s AVMA pays more for the harder step. Accelerator payments are per-dose top-ups paid on top of the UNICEF contract price, and a manufacturer receives only the highest rate that applies. <Cite s="avma" />
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-105 text-left text-sm">
              <caption className="sr-only">AVMA accelerator payment tiers</caption>
              <thead><tr className="border-b border-slate-800 font-mono text-[11px] text-slate-500"><th scope="col" className="py-2 pr-4 font-medium">Top-up</th><th scope="col" className="py-2 font-medium">What qualifies</th></tr></thead>
              <tbody>
                {AVMA_TIERS.map((r) => (
                  <tr key={r.rate} className="border-b border-slate-800/60 last:border-b-0">
                    <td className="py-2 pr-4 font-mono text-cyan-300">{r.rate}</td>
                    <td className="py-2 text-slate-300">{r.what}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="max-w-3xl text-xs leading-relaxed text-slate-400">The simulator covers both halves of that table: drug substance (upstream through UF/DF) and fill and finish.</p>
        </section>

        <section aria-labelledby="sov-workforce" className="space-y-4">
          <h2 id="sov-workforce" className="text-lg font-bold text-white">The workforce gap, and where Salus Tech fits</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-3 rounded-2xl border border-slate-800 bg-[#070d18] p-5">
              <h3 className="text-sm font-bold text-white">What the plan says</h3>
              <ul className="space-y-2 text-sm leading-relaxed text-slate-300">
                <li>• Growing from roughly 2,000–3,000 to about 12,500 vaccine staff means quadrupling the workforce, with about 10,500 people trained.</li>
                <li>• Targets for graduates of the regional capability and capacity centres: 35 % (~4,500) by 2025, 90 % (~11,500) by 2030, 100 % (~12,500) by 2040.</li>
                <li>• Drug substance skills are mostly learned inside working manufacturing facilities, and training for them is hard to find in most countries, Africa included.</li>
              </ul>
              <Cite s="ffa" where="Programme 4 and 5" />
              <p className="text-xs leading-relaxed text-slate-400">The 2026 Africa CDC / AVMI fellowship call repeats the 12,500 figure and notes a gap in drug substance and manufacturing R&amp;D skills. <Cite s="fellowship" /></p>
            </div>
            <div className="space-y-3 rounded-2xl border border-slate-800 bg-[#070d18] p-5">
              <h3 className="text-sm font-bold text-white">What a trainee practises here (our positioning)</h3>
              <ul className="space-y-1.5 text-sm text-slate-300">
                {SALUS_FIT.map((f) => (
                  <li key={f.stage} className="grid grid-cols-[130px_1fr] gap-x-3">
                    <span className="font-mono text-xs text-cyan-300">{f.stage}</span>
                    <span>{f.skill}</span>
                  </li>
                ))}
              </ul>
              <p className="text-xs leading-relaxed text-slate-400">Scope limit: the Framework prioritises seven vaccine technology platforms. This simulator models one, a recombinant protein antigen made in cell culture. mRNA, viral-vector and live-attenuated processes are not modelled.</p>
            </div>
          </div>
        </section>

        <section aria-labelledby="sov-kenya" className="space-y-3 rounded-2xl border border-slate-800 bg-[#070d18] p-6">
          <h2 id="sov-kenya" className="flex items-center gap-2 text-lg font-bold text-white"><Globe className="h-5 w-5 text-cyan-400" />Kenya BioVax Institute</h2>
          <ul className="space-y-2 text-sm leading-relaxed text-slate-300">
            <li>• State-owned and incorporated in 2021 to manufacture, package and commercialise vaccines and other health products. A 2020 feasibility study asked whether Kenya could make its own vaccines before it exits Gavi support in 2029. <Cite s="biovax" /></li>
            <li>• Working toward a Smart Vaccine Manufacturing Facility at Konza Technopolis; a delegation visited Korea in October 2025 on technology transfer and workforce training. <Cite s="biovaxKorea" /></li>
            <li>• A staff member joined a four-week biomanufacturing programme at CSIR in Pretoria in May 2025, covering fermentation engineering, bioreactor operation, chromatography and HPLC. <Cite s="biovaxTraining" /></li>
          </ul>
        </section>

        <section aria-labelledby="sov-sources" className="space-y-3">
          <h2 id="sov-sources" className="text-lg font-bold text-white">Sources</h2>
          <ul className="space-y-2 text-sm">
            {(Object.keys(SRC) as SrcKey[]).map((k) => (
              <li key={k}>
                <a href={SRC[k].url} target="_blank" rel="noreferrer noopener" className="inline-flex items-start gap-1.5 leading-snug text-slate-300 underline decoration-slate-700 underline-offset-2 hover:text-cyan-300">
                  {SRC[k].label}<ExternalLink className="mt-0.5 h-3 w-3 shrink-0 text-slate-500" aria-hidden />
                </a>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

/* ============================================================================
 * 13. MAIN APPLICATION
 * ========================================================================== */
type Tab = 'PLATFORM' | 'SCIENCE' | 'SOVEREIGNTY' | 'RESOURCES';
const TABS: { id: Tab; Icon: LucideIcon }[] = [
  { id: 'PLATFORM', Icon: Cpu },
  { id: 'SCIENCE', Icon: Activity },
  { id: 'SOVEREIGNTY', Icon: Globe },
  { id: 'RESOURCES', Icon: BookOpen },
];
const SPEEDS = [1, 4, 16];

export default function SalusDigitalTwin() {
  const sim = useProcessSimulation();
  const { snap, running, speed, upIn, dsIn } = sim;
  const { phase, ds } = snap;

  const [tab, setTab] = useState<Tab>('PLATFORM');
  const [muted, setMuted] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [split, setSplit] = useState(56); // % of the left column given to the 3D view
  const [selected, setSelected] = useState<number | null>(null);
  const [overview, setOverview] = useState(false);
  const [nonce, setNonce] = useState(0);
  const [hotspot, setHotspot] = useState<HotspotInfo | null>(null);
  const [lesson, setLesson] = useState<string | null>(null);
  const prevSplit = useRef(56);

  const activeStage = phase === 'UPSTREAM' ? 0 : 1 + (ds ? ds.stage : 0);
  const viewStage = selected ?? activeStage;
  const doneCount = phase === 'UPSTREAM' ? 0 : 1 + (ds ? ds.stage + (ds.status === 'COMPLETE' ? 1 : 0) : 0);

  // follow the process automatically whenever the active stage changes
  useEffect(() => { setSelected(null); }, [activeStage]);
  useEffect(() => { setLesson(null); }, [viewStage]);

  const alarms = useMemo(() => evaluateAlarms(phase, snap.up, upIn, ds, dsIn), [phase, snap.up, upIn, ds, dsIn]);
  const hasCritical = alarms.some((a) => a.severity === 'critical');

  useEffect(() => {
    if (hasCritical && running && !muted) audio.startAlarm();
    else audio.stopAlarm();
    return () => audio.stopAlarm();
  }, [hasCritical, running, muted]);

  const live: SceneLive = {
    focus: overview ? -1 : viewStage,
    nonce,
    visible: tab === 'PLATFORM',
    active: activeStage,
    doneCount,
    running,
    rpm: upIn.rpm,
    sparge: upIn.spargeRate,
    temperature: upIn.temperature,
    shear: shearAt(upIn.rpm),
    dissolvedOxygen: snap.up.dissolvedOxygen,
    upSeverity: phase === 'UPSTREAM' ? (hasCritical ? 2 : alarms.length ? 1 : 0) : 0,
    progress: ds?.progress ?? 0,
    pressureNorm: ds && ds.live.limit > 0 ? ds.live.pressure / ds.live.limit : 0,
    loadFrac: ds?.live.loadFrac ?? 0,
    breakthrough: ds?.live.breakthrough ?? 0,
    ph: ds?.live.ph ?? 7,
    volumeFrac: ds?.live.volumeFrac ?? 1,
    label: ds?.live.label ?? '',
  };

  const stageLocked = (stageIdx: number): boolean => {
    if (stageIdx === 0) return phase !== 'UPSTREAM';
    if (phase === 'UPSTREAM' || !ds || ds.status === 'FAILED' || ds.status === 'IDLE') return false;
    const di = stageIdx - 1;
    if (di < ds.stage) return true;
    if (di === ds.stage) return ds.progress > 0;
    return false;
  };

  const chartA = phase === 'UPSTREAM' ? UP_CHART_A : DS_CHART_A;
  const chartB = phase === 'UPSTREAM' ? UP_CHART_B : DS_CHART_B;
  const histData = phase === 'UPSTREAM' ? snap.upHist : snap.dsHist;
  const metrics = useMemo(() => buildMetrics(snap), [snap]);

  const toggleMaximise = () => {
    if (split < 80) { prevSplit.current = split; setSplit(86); }
    else setSplit(prevSplit.current);
  };

  const exportCoA = () => {
    audio.chime(980, 0.15);
    const stamp = Date.now();
    const checks = ds ? releaseChecks(ds, dsIn) : [];
    const status =
      snap.upHours <= 0 ? 'NOT RUN'
        : !ds ? 'IN PROCESS (UPSTREAM)'
          : ds.status === 'FAILED' ? 'REJECTED'
            : ds.status !== 'COMPLETE' ? 'IN PROCESS (DOWNSTREAM)'
              : checks.every((c) => c.pass) ? 'CONFORMS' : 'OUT OF SPEC';
    const coa = {
      institution: 'Salus Tech Digital Twin',
      unit: 'STR-2000L + downstream train [pilot cGMP simulation]',
      lotId: `KE-SALUS-EPI-${String(stamp).slice(-6)}`,
      timestamp: new Date(stamp).toISOString(),
      releaseStatus: status,
      upstream: {
        durationHours: Number(snap.upHours.toFixed(1)),
        finalVcd_1e6_per_mL: Number((snap.up.biomass * BIO.cellsPerGram).toFixed(2)),
        titer_g_per_L: Number(snap.up.titer.toFixed(3)),
        peakShear_Pa: Number(snap.peakShear.toFixed(2)),
        shearLimit_Pa: BIO.criticalShear,
        finalPh: Number(snap.up.brothPh.toFixed(2)),
      },
      downstream: ds && {
        status: ds.status,
        failure: ds.failure,
        overallRecovery_pct: Number(((ds.product / ds.startProduct) * 100).toFixed(1)),
        recoveryByStage_pct: ds.recovery.map((r, i) => ({ stage: STAGES[i + 1].short, recovery: Number(r.toFixed(1)) })),
        viralLrv: Number(ds.lrv.toFixed(2)),
        checks,
      },
      deviations: snap.deviations.map((d) => ({ hours: Number(d.hours.toFixed(1)), phase: d.phase, severity: d.severity, text: d.text })),
      note: 'Educational simulation. Parameters are uncalibrated placeholders; not a regulatory record.',
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(coa, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `CoA_SalusTech_${coa.lotId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const nextHint =
    phase === 'UPSTREAM'
      ? snap.upHours === 0
        ? 'Start the culture, or use Skip to jump ahead. Click the glowing nodes on the skid for sensor detail.'
        : snap.up.titer < MIN_HARVEST_TITER
          ? 'Let the culture build antigen, then harvest.'
          : 'Harvest when titer looks good (day 14 is the target) to start the downstream train.'
      : ds?.status === 'IDLE'
        ? 'Adjust the downstream settings if you like, then start. Each unit operation locks once it begins.'
        : ds?.status === 'FAILED'
          ? 'Batch rejected. Read the red alarm, change the settings, then retry from harvest.'
          : ds?.status === 'COMPLETE'
            ? 'Train complete. Check the release panel, export the CoA, or retry with different settings.'
            : 'Downstream is running. Tick “Pause between unit operations” to inspect each step.';

  /* ---------- sidebar body ---------- */
  const defs = PARAMS[viewStage];
  const locked = stageLocked(viewStage);
  const stage = STAGES[viewStage];
  const runLabel =
    running ? (phase === 'UPSTREAM' ? 'Halt' : 'Pause')
      : phase === 'UPSTREAM' ? (snap.upHours >= BIO.targetCultureH ? 'Culture complete' : snap.upHours > 0 ? 'Resume culture' : 'Start culture')
        : ds?.status === 'IDLE' ? 'Start downstream'
          : ds?.status === 'RUNNING' ? 'Resume'
            : ds?.status === 'COMPLETE' ? 'Train complete' : 'Batch rejected';
  const runDisabled = !running && ((phase === 'UPSTREAM' && snap.upHours >= BIO.targetCultureH) || (phase === 'DOWNSTREAM' && (ds?.status === 'COMPLETE' || ds?.status === 'FAILED')));

  return (
    <div className={`${fontUi.variable} ${fontData.variable} salus-root flex h-dvh w-full flex-col overflow-hidden bg-[#030712] text-slate-100`}>
      <style>{`
        .salus-root { font-family: var(--font-ui), system-ui, sans-serif; }
        .salus-root .font-mono { font-family: var(--font-data), ui-monospace, monospace; }
      `}</style>
      {/* ---------------- header ---------------- */}
      <header className="z-30 flex h-12 shrink-0 items-center gap-3 border-b border-slate-800/90 bg-[#070d18]/90 px-3 backdrop-blur sm:px-5">
        <div className="flex shrink-0 items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-linear-to-br from-cyan-400 to-emerald-500 font-mono text-sm font-black text-slate-950">⬡</div>
          <div className="leading-tight">
            <div className="font-mono text-sm font-bold tracking-wider text-white">SALUS TECH</div>
            <div className="hidden font-mono text-[9px] text-cyan-400 lg:block">cGMP digital twin · Salus Inc.</div>
          </div>
        </div>

        <nav className="mx-auto flex min-w-0 items-center gap-1 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/80 p-1">
          {TABS.map(({ id, Icon }) => (
            <button
              key={id}
              onClick={() => { audio.chime(750, 0.08); setTab(id); }}
              className={`flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1 font-mono text-xs font-semibold transition ${tab === id ? 'bg-linear-to-r from-cyan-500 to-emerald-500 text-slate-950' : 'text-slate-400 hover:bg-slate-800/60 hover:text-white'}`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{id}</span>
            </button>
          ))}
        </nav>

        <div className="flex shrink-0 items-center gap-2">
          <div className="flex overflow-hidden rounded-lg border border-slate-800" role="group" aria-label="Simulation speed">
            {SPEEDS.map((s) => (
              <button key={s} onClick={() => sim.setSpeed(s)} aria-pressed={speed === s} className={`px-2 py-1 font-mono text-[11px] ${speed === s ? 'bg-cyan-500/20 text-cyan-300' : 'bg-slate-950 text-slate-500 hover:text-slate-300'}`}>{s}×</button>
            ))}
          </div>
          <button
            onClick={() => { const m = !muted; setMuted(m); audio.setMuted(m); }}
            title={muted ? 'Unmute alarms' : 'Mute alarms'}
            aria-label={muted ? 'Unmute alarms' : 'Mute alarms'}
            className={`rounded-lg border p-1.5 ${muted ? 'border-slate-800 bg-slate-950 text-slate-500' : 'border-cyan-500/40 bg-cyan-950/30 text-cyan-400'}`}
          >
            {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>
          <button onClick={exportCoA} className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-1.5 font-mono text-xs text-cyan-400 hover:border-cyan-500/50">
            <Download className="h-3.5 w-3.5" />
            <span className="hidden md:inline">Export CoA</span>
          </button>
        </div>
      </header>

      <main className="relative min-h-0 flex-1 overflow-hidden">
        {(
          <div
            className={tab === 'PLATFORM' ? 'grid h-full min-h-0 gap-3 p-3' : 'hidden'}
            style={{ gridTemplateColumns: sidebarOpen ? 'minmax(0,1fr) clamp(290px, 26vw, 400px)' : 'minmax(0,1fr) 44px' }}
          >
            {/* ================= left column: train bar / 3D / data dock ================= */}
            <div
              className="grid min-h-0 min-w-0 gap-3"
              style={{ gridTemplateRows: `auto minmax(160px, ${split}fr) minmax(120px, ${100 - split}fr)` }}
            >
              {/* process train bar + alarms */}
              <section style={{ gridRow: 1 }} className="min-w-0 rounded-2xl border border-slate-800/90 bg-[#070d18] p-2">
                <div className="flex gap-1.5 overflow-x-auto pb-0.5" role="tablist" aria-label="Process train">
                  {STAGES.map((st, i) => {
                    const failed = phase === 'DOWNSTREAM' && ds?.status === 'FAILED' && i === activeStage;
                    const done = i < doneCount;
                    const isActive = i === activeStage;
                    const isView = i === viewStage;
                    const cls = failed ? 'border-rose-600 bg-rose-950/60 text-rose-200'
                      : isActive ? 'border-cyan-400 bg-cyan-500/15 text-cyan-200'
                        : done ? 'border-emerald-700/70 bg-emerald-950/30 text-emerald-300'
                          : 'border-slate-800 bg-slate-950 text-slate-500';
                    return (
                      <button
                        key={st.key}
                        role="tab"
                        aria-selected={isView}
                        onClick={() => { setSelected(i); setOverview(false); audio.chime(700, 0.06); }}
                        className={`flex min-w-23 flex-1 items-center justify-center gap-1.5 rounded-lg border px-2 py-1.5 font-mono text-[11px] font-semibold transition ${cls} ${isView ? 'ring-1 ring-white/40' : ''}`}
                      >
                        {done && !isActive ? <Check className="h-3.5 w-3.5 shrink-0" /> : <st.Icon className="h-3.5 w-3.5 shrink-0" />}
                        <span className="truncate">{st.short}</span>
                      </button>
                    );
                  })}
                </div>
                {alarms.length > 0 && (
                  <div className="mt-1.5 flex flex-col gap-1" aria-live="polite">
                    {alarms.map((a) => (
                      <div key={a.id} className={`flex items-start gap-2 rounded-lg border px-2.5 py-1 font-mono text-[11px] ${a.severity === 'critical' ? 'border-rose-700 bg-rose-950/70 text-rose-200' : 'border-amber-700/70 bg-amber-950/50 text-amber-200'}`}>
                        <AlertOctagon className={`mt-px h-3.5 w-3.5 shrink-0 ${a.severity === 'critical' ? 'text-rose-400' : 'text-amber-400'}`} />
                        <span>{a.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              {/* 3D viewport */}
              <section style={{ gridRow: 2 }} className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border border-slate-800/90 bg-[#030712]">
                <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-800/80 bg-[#070d18] px-3 py-1.5 font-mono text-[11px]">
                  <span className="flex items-center gap-1.5 font-bold text-cyan-400"><span className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />DIGITAL TWIN</span>
                  <span className="hidden text-slate-500 xl:inline">Drag: orbit · Wheel: zoom · Right-drag: pan · Click nodes</span>
                  <div className="ml-auto flex items-center gap-2">
                    <label className="flex items-center gap-1.5 text-slate-400">
                      <span className="hidden sm:inline">3D / data</span>
                      <input type="range" min={25} max={86} value={split} onChange={(e) => setSplit(Number(e.target.value))} aria-label="3D view height" className="h-1 w-20 cursor-pointer appearance-none rounded bg-slate-800 accent-cyan-400 sm:w-28" />
                    </label>
                    <button onClick={() => setOverview((v) => !v)} aria-pressed={overview} className={`flex items-center gap-1 rounded border px-2 py-0.5 ${overview ? 'border-cyan-500 bg-cyan-500/15 text-cyan-300' : 'border-slate-700 text-slate-400 hover:text-white'}`}>
                      <Layers className="h-3 w-3" /> Whole train
                    </button>
                    <button onClick={() => setNonce((n) => n + 1)} title="Re-fit camera" aria-label="Re-fit camera" className="rounded border border-slate-700 p-1 text-slate-400 hover:text-white"><Scan className="h-3 w-3" /></button>
                    <button onClick={toggleMaximise} title={split < 80 ? 'Expand 3D view' : 'Restore'} aria-label="Expand or restore 3D view" className="rounded border border-slate-700 p-1 text-slate-400 hover:text-white">
                      {split < 80 ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
                    </button>
                  </div>
                </div>

                <div className="relative min-h-0 flex-1">
                  <ProcessTrainScene live={live} onSelect={(h) => { setHotspot(h); setSelected(h.stage); setOverview(false); }} />
                </div>

                <div className="flex min-h-9 shrink-0 items-start gap-2 border-t border-slate-800/80 bg-[#070d18] px-3 py-1.5 text-xs">
                  {hotspot ? (
                    <>
                      <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-cyan-400" />
                      <p className="min-w-0 flex-1 leading-snug text-slate-300"><span className="font-bold text-white">{hotspot.name}.</span> {hotspot.desc}</p>
                      <button onClick={() => setHotspot(null)} aria-label="Dismiss" className="shrink-0 text-slate-500 hover:text-white"><X className="h-3.5 w-3.5" /></button>
                    </>
                  ) : (
                    <p className="font-mono text-[11px] text-slate-400">{nextHint}</p>
                  )}
                </div>
              </section>

              {/* data dock */}
              <section style={{ gridRow: 3 }} className="flex min-h-0 min-w-0 flex-col gap-3 overflow-y-auto rounded-2xl border border-slate-800/90 bg-[#070d18] p-3">
                <div className="grid shrink-0 gap-2 grid-cols-[repeat(auto-fill,minmax(128px,1fr))]">
                  {metrics.map((m) => <Metric key={m.label} {...m} />)}
                </div>
                <div className="grid shrink-0 gap-3 grid-cols-[repeat(auto-fit,minmax(280px,1fr))]" style={{ height: 'clamp(160px, 26vh, 280px)' }}>
                  <LiveChart title={phase === 'UPSTREAM' ? 'Culture' : 'Unit operation'} channels={chartA} data={histData} />
                  <LiveChart title={phase === 'UPSTREAM' ? 'Gas & quality' : 'Pressure & clearance'} channels={chartB} data={histData} />
                </div>
                {ds && <ReleasePanel ds={ds} dsIn={dsIn} />}
                {snap.deviations.length > 0 && (
                  <div className="shrink-0 rounded-xl border border-slate-800 bg-slate-950/70 p-3">
                    <h3 className="mb-1.5 font-mono text-xs font-bold text-white">Event log</h3>
                    <ul className="max-h-32 space-y-0.5 overflow-y-auto font-mono text-[11px]">
                      {snap.deviations.slice().reverse().map((d) => (
                        <li key={d.id} className={d.severity === 'critical' ? 'text-rose-300' : d.severity === 'warning' ? 'text-amber-300' : 'text-slate-400'}>
                          <span className="text-slate-600">{d.phase === 'UPSTREAM' ? 'USP' : 'DSP'} {d.hours.toFixed(1)} h</span> {d.text}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </div>

            {/* ================= right column: controls ================= */}
            <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border border-slate-800/90 bg-[#070d18]">
              {!sidebarOpen ? (
                <button onClick={() => setSidebarOpen(true)} aria-label="Open controls" className="flex h-full w-full items-start justify-center pt-3 text-slate-400 hover:text-white"><PanelRightOpen className="h-5 w-5" /></button>
              ) : (
                <>
                  <div className="flex shrink-0 items-start justify-between gap-2 border-b border-slate-800 p-3">
                    <div className="min-w-0">
                      <h2 className="flex items-center gap-2 text-sm font-bold text-white"><Sliders className="h-4 w-4 shrink-0 text-cyan-400" /><span className="truncate">{stage.label}</span></h2>
                      <p className="mt-1 text-[11px] leading-snug text-slate-400">{stage.blurb}</p>
                    </div>
                    <div className="flex shrink-0 flex-col items-end gap-1.5">
                      <button onClick={() => setSidebarOpen(false)} aria-label="Collapse controls" className="text-slate-500 hover:text-white"><PanelRightClose className="h-4 w-4" /></button>
                      <span className={`rounded-full border px-2 py-0.5 font-mono text-[10px] font-bold ${running ? 'border-emerald-800 bg-emerald-950 text-emerald-400' : 'border-slate-800 bg-slate-900 text-slate-500'}`}>{running ? 'RUNNING' : 'STANDBY'}</span>
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
                    {locked && <p className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 font-mono text-[11px] text-slate-400">{viewStage === 0 ? 'Culture is harvested: upstream settings are locked.' : 'This unit operation has started or finished: settings are locked. Retry from harvest to change them.'}</p>}
                    <div className="grid gap-2 grid-cols-[repeat(auto-fit,minmax(118px,1fr))]">
                      {defs.map((p) => {
                        const value = (p.group === 'up' ? (upIn as unknown as Record<string, number>) : (dsIn as unknown as Record<string, number>))[p.key];
                        return (
                          <RotaryDial
                            key={`${p.group}:${p.key}`}
                            id={`${p.group}:${p.key}`}
                            label={p.label}
                            unit={p.unit}
                            min={p.min}
                            max={p.max}
                            step={p.step}
                            safeMin={p.safeMin}
                            safeMax={p.safeMax}
                            color={p.color}
                            value={value}
                            disabled={locked}
                            onChange={sim.setParam}
                          />
                        );
                      })}
                    </div>

                    <div className="space-y-1.5 border-t border-slate-800/80 pt-3">
                      <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-400"><ShieldAlert className="h-3.5 w-3.5 text-amber-400" />Teaching scenarios</div>
                      <div className="grid gap-2 font-mono text-[11px] grid-cols-[repeat(auto-fit,minmax(110px,1fr))]">
                        {SCENARIOS[viewStage].map((sc) => (
                          <button
                            key={sc.label}
                            disabled={locked}
                            onClick={() => { audio.chime(440, 0.2); sim.applyParams(sc.up, sc.ds); setLesson(sc.lesson); }}
                            className={`rounded-lg border border-slate-800 bg-slate-950 p-2 hover:border-current disabled:opacity-40 ${SCENARIO_TONE[sc.tone]}`}
                          >
                            {sc.label}
                          </button>
                        ))}
                        <button disabled={locked} onClick={() => { sim.restoreNominal(viewStage); setLesson(null); }} className="rounded-lg border border-slate-800 bg-slate-950 p-2 text-slate-300 hover:border-slate-500 disabled:opacity-40">Back to nominal</button>
                      </div>
                      {lesson && <p className="rounded-lg border border-slate-800 bg-slate-950 px-2.5 py-1.5 text-[11px] leading-snug text-slate-300">{lesson}</p>}
                    </div>
                  </div>

                  <div className="shrink-0 space-y-2 border-t border-slate-800 p-3">
                    <div className="flex gap-2">
                      <button
                        disabled={runDisabled}
                        onClick={() => { audio.chime(running ? 380 : 880, 0.15); sim.toggleRun(); }}
                        className={`flex flex-1 items-center justify-center gap-2 rounded-xl py-2.5 font-mono text-xs font-bold transition disabled:cursor-not-allowed disabled:opacity-50 ${running ? 'bg-rose-600 text-white hover:bg-rose-500' : 'bg-linear-to-r from-cyan-500 to-emerald-400 text-slate-950 hover:brightness-110'}`}
                      >
                        {running ? <Square className="h-4 w-4 fill-current" /> : <Play className="h-4 w-4 fill-current" />}
                        {runLabel}
                      </button>
                      <button onClick={() => { audio.chime(500, 0.1); sim.reset(); setSelected(null); setHotspot(null); setOverview(false); setLesson(null); }} title="Reset the whole batch" aria-label="Reset the whole batch" className="rounded-xl border border-slate-800 bg-slate-900 p-2.5 text-slate-300 hover:bg-slate-800"><RotateCcw className="h-4 w-4" /></button>
                    </div>

                    {phase === 'UPSTREAM' && (
                      <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                        <button disabled={running || snap.upHours >= BIO.targetCultureH} onClick={() => sim.fastForward(48)} className="rounded-lg border border-slate-800 bg-slate-950 py-1.5 text-slate-300 hover:border-cyan-500/60 disabled:opacity-40">Skip +2 days</button>
                        <button disabled={running || snap.upHours >= BIO.targetCultureH} onClick={() => sim.fastForward(BIO.targetCultureH)} className="rounded-lg border border-slate-800 bg-slate-950 py-1.5 text-slate-300 hover:border-cyan-500/60 disabled:opacity-40">Skip to day 14</button>
                      </div>
                    )}
                    {phase === 'UPSTREAM' ? (
                      <button
                        onClick={() => { audio.chime(980, 0.15); sim.harvest(); }}
                        disabled={snap.up.titer < MIN_HARVEST_TITER}
                        className="flex w-full items-center justify-center gap-2 rounded-xl border border-emerald-700 bg-emerald-950/50 py-2 font-mono text-xs font-bold text-emerald-300 hover:bg-emerald-900/50 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        <Wheat className="h-4 w-4" /> Harvest and start downstream
                      </button>
                    ) : (
                      <>
                        {(ds?.status === 'FAILED' || ds?.status === 'COMPLETE') && (
                          <button onClick={() => sim.retryDownstream()} className="flex w-full items-center justify-center gap-2 rounded-xl border border-amber-700 bg-amber-950/40 py-2 font-mono text-xs font-bold text-amber-300 hover:bg-amber-900/40">
                            <RotateCcw className="h-4 w-4" /> Retry downstream from harvest
                          </button>
                        )}
                        <label className="flex cursor-pointer items-center gap-2 font-mono text-[11px] text-slate-400">
                          <input type="checkbox" checked={sim.pauseBetween} onChange={(e) => sim.setPauseBetween(e.target.checked)} className="accent-cyan-400" />
                          Pause between unit operations
                        </label>
                      </>
                    )}
                    <div className="text-center font-mono text-[10px] text-slate-600">Fixed-step RK4 engine · {SIM.dt} h/step · {speed}× speed</div>
                  </div>
                </>
              )}
            </aside>
          </div>
        )}

        {/* ---------------- SCIENCE ---------------- */}
        {tab === 'SCIENCE' && (
          <div className="h-full overflow-y-auto">
            <div className="mx-auto max-w-5xl space-y-8 p-6 md:p-10">
              <header className="space-y-2 border-b border-slate-800 pb-5">
                <span className="rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 font-mono text-xs text-cyan-400">Mathematical foundations · ICH Q8(R2)</span>
                <h1 className="pt-2 text-3xl font-extrabold text-white">Mechanistic models, upstream to fill</h1>
                <p className="max-w-3xl text-sm leading-relaxed text-slate-400">The upstream culture is a set of coupled ODEs integrated with a classical fourth-order Runge-Kutta scheme at a fixed 0.08 h step. Downstream unit operations are mass-balanced stage models that hand antigen, impurities and purity to the next stage. Every coefficient is a placeholder calibration for teaching.</p>
              </header>
              <div className="grid gap-5 md:grid-cols-2">
                {SCIENCE_CARDS.map((c) => (
                  <article key={c.id} className="space-y-3 rounded-2xl border border-slate-800 bg-[#070d18] p-5">
                    <div className={`flex items-center justify-between font-mono text-xs font-bold ${c.tone}`}><span>{c.title}</span><span>{c.sym}</span></div>
                    <div className="space-y-1 overflow-x-auto rounded-xl border border-slate-800/80 bg-slate-950 p-3 font-mono text-xs text-slate-200">
                      {c.lines.map((l) => <div key={l} className="whitespace-nowrap">{l}</div>)}
                    </div>
                    <p className="text-xs leading-relaxed text-slate-400">{c.body}</p>
                  </article>
                ))}
              </div>
              <article className="space-y-3 rounded-2xl border border-cyan-900/40 bg-slate-950 p-5 font-mono text-xs text-slate-300">
                <h3 className="flex items-center gap-2 text-sm font-bold text-white"><Cpu className="h-4 w-4 text-cyan-400" />Runge-Kutta 4th order</h3>
                <div className="space-y-1 overflow-x-auto rounded-xl bg-slate-900/90 p-4 text-cyan-300">
                  <div>k1 = f(yₙ)</div><div>k2 = f(yₙ + dt·k1/2)</div><div>k3 = f(yₙ + dt·k2/2)</div><div>k4 = f(yₙ + dt·k3)</div>
                  <div className="pt-1 font-bold text-emerald-400">yₙ₊₁ = yₙ + (dt/6)(k1 + 2k2 + 2k3 + k4)</div>
                </div>
                <p className="text-slate-500">Thresholds in the lysis and aggregation terms are non-smooth, so the effective order drops near them. The system has no explicit time dependence, so each stage evaluates f at the stage state only.</p>
              </article>
            </div>
          </div>
        )}

        {/* ---------------- SOVEREIGNTY ---------------- */}
        {tab === 'SOVEREIGNTY' && <SovereigntyTab />}

        {/* ---------------- RESOURCES ---------------- */}
        {tab === 'RESOURCES' && (
          <div className="h-full overflow-y-auto">
            <div className="mx-auto max-w-5xl space-y-8 p-6 md:p-10">
              <header className="space-y-2 border-b border-slate-800 pb-5">
                <span className="rounded-full border border-cyan-800 bg-cyan-950/60 px-3 py-1 font-mono text-xs text-cyan-400">Regulatory standards and literature</span>
                <h1 className="pt-2 text-3xl font-extrabold text-white">Standards and references</h1>
              </header>
              <div className="grid gap-5 md:grid-cols-3">
                {[
                  { code: 'ICH Q8(R2)', tone: 'text-emerald-400', title: 'Pharmaceutical development', text: 'Design space: the multivariate region of process parameters that assures quality. The simulator’s alarm bands are univariate normal operating ranges, not a design space.' },
                  { code: 'ICH Q9', tone: 'text-amber-400', title: 'Quality risk management', text: 'FMEA-style thinking for thermal drift, shear lysis, filter blockage and failed viral clearance.' },
                  { code: 'ICH Q10', tone: 'text-cyan-400', title: 'Pharmaceutical quality system', text: 'Deviations, CAPA and release decisions. The event log and CoA export are a teaching stand-in only.' },
                ].map((r) => (
                  <article key={r.code} className="space-y-2 rounded-2xl border border-slate-800 bg-[#070d18] p-5">
                    <div className={`font-mono text-xs font-bold ${r.tone}`}>{r.code}</div>
                    <h3 className="text-sm font-bold text-white">{r.title}</h3>
                    <p className="text-xs leading-relaxed text-slate-400">{r.text}</p>
                  </article>
                ))}
              </div>
              <article className="space-y-3 rounded-2xl border border-slate-800 bg-[#070d18] p-6">
                <h3 className="flex items-center gap-2 font-mono text-base font-bold text-white"><BookOpen className="h-4 w-4 text-cyan-400" />Citations (APA 7th)</h3>
                <ul className="space-y-3 border-t border-slate-800 pt-4 font-mono text-xs leading-relaxed text-slate-400">
                  {CITATIONS.map((c) => (
                    <li key={c.text}>• {c.text}{c.url && <> <a href={c.url} target="_blank" rel="noreferrer noopener" className="text-cyan-400 underline decoration-cyan-800 underline-offset-2 hover:text-cyan-300">link</a></>}{!c.verified && <span className="ml-1.5 rounded border border-amber-700/60 bg-amber-950/60 px-1 text-[9px] text-amber-300">verify by DOI</span>}</li>
                  ))}
                </ul>
              </article>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
