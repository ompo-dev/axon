import { expect, test } from 'bun:test';
import { label, latency, preserveObservedAlgorithmEvidence } from '../src/useAxon';
import type { Snapshot } from '../src/types';
import { workflows } from '../laboratory.mjs';

test('sub-millisecond measurements retain precision and unknown values stay unknown', () => {
  expect(latency(0)).toBe('0 ns');
  expect(latency(250)).toBe('250 ns');
  expect(latency(12000)).toBe('12 us');
  expect(latency(1234000)).toBe('1,23 ms');
  for (const ns of [undefined, NaN, -1, Infinity]) expect(latency(ns)).toBe('--');
});

test('every executable workflow has a usable display label', () => {
  for (const workflow of workflows) {
    expect(label[workflow.id as keyof typeof label]?.trim().length).toBeGreaterThan(0);
  }
  expect(label.structural.toUpperCase()).toContain('ESTRUTURAS');
});

test('older snapshots preserve only observed algorithm evidence of the same run and session', () => {
  const snapshot = (session = 'one', id = 'run', lastCoreSeq = 10) =>
    ({
      session,
      current: { id, trace: { lastCoreSeq } },
    }) as Snapshot;
  const previous = snapshot();
  previous.current!.algorithmInvestigation = {
    event: 'algorithm_investigation',
    seq: 5,
    session: 'one',
    time: 1,
    run: 'run',
    coreSeq: 5,
    templates: 3,
  };
  const compatible = snapshot();
  previous.current!.algorithmInquiry = {
    ...previous.current!.algorithmInvestigation,
    phase: 'query_observed',
    query_count: 1,
    last_inquiry: '{"input":[2,3],"actual":5}',
  };
  preserveObservedAlgorithmEvidence(compatible, previous);
  expect(compatible.current!.algorithmInvestigation?.templates).toBe(3);
  expect(compatible.current!.algorithmInquiry?.query_count).toBe(1);
  for (const incompatible of [
    snapshot('two'),
    snapshot('one', 'other'),
    snapshot('one', 'run', 4),
  ]) {
    preserveObservedAlgorithmEvidence(incompatible, previous);
    expect(incompatible.current!.algorithmInvestigation).toBeUndefined();
    expect(incompatible.current!.algorithmInquiry).toBeUndefined();
  }
});
