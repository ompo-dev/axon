const exact = n => {
  const x = Number(n);
  if (!Number.isSafeInteger(x)) throw Error('M0 auditor exact integer range');
  return x;
};
export function m0Reference(family, values) {
  if (family === 'state-source-wrap') return [values];
  if (family === 'state-source-initial') return [values, []];
  if (family === 'state-source-prepend') return [values[1]].concat(values[0]);
  if (family === 'state-source-history') {
    const total = exact(BigInt(values[0][0]) + BigInt(values[1]));
    return [total,values[0][1].concat(total)];
  }
  if (family === 'state-source-modular') {
    const total = BigInt(values[0]) + BigInt(values[1][0]), modulus = BigInt(values[1][1]);
    return exact(total - (total / modulus) * modulus);
  }
  if (family === 'state-source-total') return [exact(BigInt(values[0])+BigInt(values[1][0])),values[1][1]];
  if (family === 'state-source-remainder') {
    const a = BigInt(values[0]), b = BigInt(values[1]);
    return exact(a - (a / b) * b);
  }
  if (family === 'order-reversal') return values.map((_,i) => values[values.length - 1 - i]);
  if (family === 'cumulative-state-history') {
    const prefix = values.map((_,i) => exact(values.slice(0,i+1).reduce((s,n) => s+BigInt(n),0n)));
    return [prefix.at(-1) ?? 0,prefix];
  }
  if (family === 'modulus-changing-state') {
    let total = 0n;
    for (const [input, modulus] of values) {
      const next = total + BigInt(input), m = BigInt(modulus);
      total = next - (next / m) * m;
    }
    return exact(total);
  }
  if (!Array.isArray(values) || values.some(x => !Number.isSafeInteger(x))) throw Error('M0 auditor input domain');
  const x = values.map(BigInt);
  if (family === 'transition-source-0') return exact(2n * x[0] + x[1]);
  if (family === 'transition-source-1') return exact(x[0] ** 2n + x[1]);
  if (family === 'transition-source-2') return exact(x[0] * x[1] + x[1] ** 2n);
  // The two closed-form references do not execute the learner's fold or teacher's recurrence.
  if (family === 'linear-recurrence') return exact(x.reduce((s, n, i) => s + n * (2n ** BigInt(x.length - i - 1)), 0n));
  if (family === 'bilinear-recurrence') return exact(x.reduce((s, n, i) => s + n * n * x.slice(i + 1).reduce((p, v) => p * v, 1n), 0n));
  if (family === 'quadratic-recurrence') {
    let state = 0n;
    for (const input of x) state = state ** 2n + input;
    return exact(state);
  }
  throw Error(`Unknown M0 reference ${family}`);
}
