import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { PNG } from 'pngjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import type { ServerResponse } from 'node:http';
import { createObservatory } from '../server.mjs';
import type { RecordEvent } from '../src/types';

const qa = path.resolve('../.axon-memory/observatory/qa/react-workspace');
async function ready(page: Page) {
  await page.goto('/');
  await expect(page.getByText('Nucleo conectado', { exact: true })).toBeAttached();
  await expect(page.getByRole('heading', { name: 'Runtime', exact: true })).toBeVisible();
  await fs.mkdir(qa, { recursive: true });
}
async function noOverflow(page: Page) {
  const result = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(result.scroll).toBeLessThanOrEqual(result.width + 1);
}
async function settledTheme(page: Page) {
  await page.evaluate(async () => {
    const animations = document
      .getAnimations()
      .filter(
        (a) => a.playState === 'running' && Number.isFinite(a.effect?.getComputedTiming().endTime),
      );
    await Promise.all(animations.map((a) => a.finished.catch(() => {})));
  });
}
async function pixels(page: Page, selector: string) {
  const image = PNG.sync.read(await page.locator(selector).screenshot());
  const colors = new Set<number>();
  for (let i = 0; i < image.data.length; i += 16)
    colors.add((image.data[i] << 16) | (image.data[i + 1] << 8) | image.data[i + 2]);
  expect(colors.size).toBeGreaterThan(25);
}
test('desktop: real snapshot, navigation, preferences, panels, source inspection and historical campaigns', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Documento mestre', exact: true }).click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('AXON: Documento Mestre');
  await expect(page.getByRole('dialog').locator('pre')).toContainText('E33');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  await noOverflow(page);
  await expect(page.locator('.matrix-row')).toHaveCount(5);
  await pixels(page, '.core-diagram');
  await page.screenshot({ path: path.join(qa, 'desktop-light.png'), fullPage: true });
  await page.getByRole('button', { name: 'Dados de Modelos', exact: true }).click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('edges');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  await page.getByRole('button', { name: 'Ocultar Recursos', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Recursos', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Paineis', exact: true }).click();
  await page.getByRole('dialog').getByLabel('Recursos').check();
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Recursos', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.screenshot({ path: path.join(qa, 'desktop-dark.png'), fullPage: true });
  await page.getByRole('button', { name: 'Tema claro', exact: true }).click();
  await page.getByRole('button', { name: 'Recolher navegacao', exact: true }).click();
  await expect(page.locator('.sidebar')).toHaveClass(/collapsed/);
  await page.locator('.brand-mark').click();
  await expect(page.locator('.sidebar')).not.toHaveClass(/collapsed/);
  await page.getByRole('button', { name: 'Rede de estados', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'Rede de estados', exact: true }).first(),
  ).toBeVisible();
  await pixels(page, '.network-svg');
  await page.getByRole('button', { name: 'Afastar grafo', exact: true }).click();
  await page.getByRole('button', { name: 'Afastar grafo', exact: true }).click();
  await expect(page.locator('.graph-controls output')).toHaveText('70%');
  await page.screenshot({ path: path.join(qa, 'network-zoom.png'), fullPage: true });
  await page
    .getByRole('button', { name: /Inspecionar .*:s0/ })
    .first()
    .click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('transitions');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  await page.getByRole('button', { name: 'Seeds e evolucao', exact: true }).click();
  await page.getByLabel('Selecionar campanha').selectOption('reserved-672804213-hybrid-v1.json');
  await expect(page.locator('.evolution-row')).toHaveCount(7);
  await expect(page.locator('.evolution-summary')).toContainText('70.000 CORRETOS');
  await page.screenshot({ path: path.join(qa, 'evolution.png'), fullPage: true });
  await page.locator('.evolution-row').nth(4).click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('sqrt');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  await page.getByRole('button', { name: 'Arquivos', exact: true }).click();
  await page
    .getByRole('textbox', { name: 'Buscar arquivos', exact: true })
    .fill('numeric_evolution.rs');
  await expect(page.locator('.file-table-row')).toHaveCount(1);
  await page.locator('.file-table-row').click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('EvolvedSkill');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  expect(errors).toEqual([]);
});

test('real Rust run through the UI: pause, checkpoint, resume, exact audit and lossless export', async ({
  page,
  request,
}) => {
  await ready(page);
  await page.getByRole('button', { name: 'Nova execucao', exact: true }).first().click();
  await page.getByRole('dialog').getByLabel('Iniciar pausado').check();
  await page.getByRole('dialog').getByRole('button', { name: 'Executar', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Avancar um checkpoint', exact: true }),
  ).toBeEnabled();
  await expect(page.locator('.runtime-row').first()).toContainText('PAUSED');
  const before = await (await request.get('/api/state')).json();
  expect(before.current.counters.queryCalls).toBe(0);
  await page.getByRole('button', { name: 'Avancar um checkpoint', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.counters.queryCalls)
    .toBe(1);
  await page.getByRole('button', { name: 'Retomar', exact: true }).click();
  await expect
    .poll(
      async () => {
        const current = (await (await request.get('/api/state')).json()).current;
        return `${current.mode}:${current.status}`;
      },
      {
        timeout: 60000,
      },
    )
    .toBe('transfer:complete');
  await expect(page.locator('.three-metrics')).toContainText('Exata');
  await expect(page.locator('.route-pulse')).toHaveCount(0);
  const state = await (await request.get('/api/state')).json();
  const records = (await (await request.get(`/api/export?id=${state.current.id}`)).text())
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  expect(state.current.result.exact).toBe(true);
  expect(state.current.result.residual).toBe(0);
  expect(state.current.counters.queryCalls).toBe(records.filter((e) => e.event === 'query').length);
  expect(state.current.trace.records).toBe(records.length);
  const core = records.filter((e) => e.coreSeq != null);
  core.forEach((e, i) => expect(e.coreSeq).toBe(i + 1));
  await page.screenshot({ path: path.join(qa, 'real-transfer-result.png'), fullPage: true });
  await page.getByRole('button', { name: 'Execucoes', exact: true }).click();
  await expect(page.locator('.event-table-row')).not.toHaveCount(0);
  await page.getByRole('button', { name: 'RESULTADOS', exact: true }).click();
  await expect(page.locator('.event-table')).toContainText('result');
});

test('mobile: responsive layout, navigation and all operational views', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'mobile-light.png'), fullPage: true });
  for (const name of [
    'Rede de estados',
    'Seeds e evolucao',
    'Laboratorio',
    'Arquivos',
    'Execucoes',
    'Conhecimento',
  ]) {
    await page.getByRole('button', { name: 'Abrir navegacao', exact: true }).click();
    await page.locator('.sidebar').getByRole('button', { name, exact: true }).click();
    await expect(page.getByRole('heading', { level: 1, name, exact: true })).toBeVisible();
    await noOverflow(page);
    await page.screenshot({
      path: path.join(qa, `mobile-${name.split(' ')[0].toLowerCase()}.png`),
      fullPage: true,
    });
  }
  await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'mobile-dark.png'), fullPage: true });
  for (const width of [320, 375, 768, 1024]) {
    await page.setViewportSize({ width, height: 900 });
    await noOverflow(page);
  }
  await page.setViewportSize({ width: 320, height: 800 });
  await page.getByRole('button', { name: 'Abrir navegacao', exact: true }).click();
  await page.locator('.sidebar').getByRole('button', { name: 'Visao geral', exact: true }).click();
  await noOverflow(page);
});

test('keyboard and accessibility: dialogs, light and dark workspace', async ({ page }) => {
  await ready(page);
  await page.keyboard.press('Control+k');
  await expect(page.getByRole('dialog', { name: 'Buscar no workspace' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark')
      await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    for (const name of [
      'Visao geral',
      'Rede de estados',
      'Seeds e evolucao',
      'Laboratorio',
      'Arquivos',
      'Execucoes',
      'Conhecimento',
    ]) {
      await page.locator('.sidebar').getByRole('button', { name, exact: true }).click();
      if (await page.locator('.widget').count())
        await expect(page.locator('.widget').first()).toHaveCSS('opacity', '1');
      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
        .analyze();
      expect(
        results.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })),
      ).toEqual([]);
    }
  }
});
test('laboratory: real program evolution, genomic inspection, artifact export and test output', async ({
  page,
  request,
}) => {
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.status, {
      timeout: 60000,
    })
    .toBe('complete');
  const state = await (await request.get('/api/state')).json();
  expect(state.current.mode).toBe('evolution');
  expect(state.current.result.passed).toBe(8);
  expect(state.current.result.correct).toBe(8000);
  expect(state.current.trace.gaps).toBe(0);
  await expect(page.getByRole('heading', { name: 'Genoma adquirido', exact: true })).toBeVisible();
  await expect(page.locator('.lab-expression code')).toContainText('sqrt');
  await pixels(page, '.genome-scroll');
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'laboratory-genome-desktop.png'), fullPage: true });
  await page.getByRole('button', { name: /Sinal 0,/ }).click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('gene');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  const artifact = await request.get(`/api/artifact?id=${state.current.id}&name=results.csv`);
  expect(artifact.ok()).toBe(true);
  expect(await artifact.text()).toContain('Admitted');
  const records = (await (await request.get(`/api/export?id=${state.current.id}`)).text())
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  expect(records.length).toBe(state.current.trace.records);
  expect(records.filter((e) => e.event === 'genome')).toHaveLength(8);
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark')
      await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    await expect(page.locator('.widget').first()).toHaveCSS('opacity', '1');
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(
      results.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })),
    ).toEqual([]);
  }
  await page.setViewportSize({ width: 320, height: 900 });
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'laboratory-genome-mobile.png'), fullPage: true });
  await page.getByLabel('Experimento', { exact: true }).selectOption('tests');
  await page.getByLabel('Suite', { exact: true }).selectOption('ui');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.status)
    .toBe('complete');
  await expect(page.locator('.lab-output')).toContainText('pass');
  await expect(page.getByRole('heading', { name: 'Genoma adquirido', exact: true })).toHaveCount(0);
  await noOverflow(page);
});

test('heritable seeds: observed population, lineage, real DNA exports and responsive inspection', async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('seed');
  await expect(page.getByLabel('Seed aleatoria', { exact: true })).toHaveValue('70143863');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(
      async () => {
        const current = (await (await request.get('/api/state')).json()).current;
        return `${current.mode}:${current.status}`;
      },
      { timeout: 60000 },
    )
    .toBe('seed:complete');
  const state = await (await request.get('/api/state')).json();
  expect(state.current.mode).toBe('seed');
  expect(state.current.candidates).toHaveLength(24);
  expect(state.current.result.solved).toBe(5);
  expect(state.current.result.correct).toBe(state.current.result.heldout);
  expect(state.current.trace.gaps).toBe(0);
  await expect(
    page.getByRole('heading', { name: 'Populacao e descendencia', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('region', { name: 'Populacao de seeds' }).locator('tbody tr'),
  ).toHaveCount(24);
  await page.getByRole('button', { name: 'Inspecionar candidato 0', exact: true }).click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('parent_a');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  const artifact = await request.get(`/api/artifact?id=${state.current.id}&name=candidate-0.axdna`);
  expect(artifact.ok()).toBe(true);
  expect((await artifact.body()).subarray(0, 8).toString()).toBe('AXDNA001');
  const revisions = page.locator('.lab-revisions');
  await expect(revisions).toHaveCount(1);
  await expect(revisions).not.toHaveAttribute('open');
  await revisions.locator('summary').click();
  await expect(revisions.getByRole('link').first()).toBeVisible();
  await noOverflow(page);
  await revisions.locator('summary').click();
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'heritable-seeds-desktop.png'), fullPage: true });
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark')
      await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    await expect(page.locator('.widget').first()).toHaveCSS('opacity', '1');
    const result = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(
      result.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })),
    ).toEqual([]);
  }
  await page.setViewportSize({ width: 320, height: 900 });
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'heritable-seeds-mobile.png'), fullPage: true });
  expect(errors).toEqual([]);
});
test('goal controller: real compositions, recorded drift, second-process results and compact views', async ({
  page,
  request,
}) => {
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('controller');
  await expect(page.getByLabel('Seed aleatoria', { exact: true })).toHaveValue('72918463');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => {
      const current = (await (await request.get('/api/state')).json()).current;
      return `${current.mode}:${current.status}`;
    })
    .toBe('controller:complete');
  const state = (await (await request.get('/api/state')).json()).current;
  expect(state.result.correct).toBe(2000);
  expect(state.result.physical_correct).toBe(1999);
  expect(state.result.drifts).toBe(1);
  expect(state.result.baseline_correct).toBeLessThan(2000);
  expect(state.trace.gaps).toBe(0);
  await expect(
    page.getByRole('heading', { name: 'Procedimento aprendido', exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole('list', { name: 'Etapas do procedimento adquirido' }).getByRole('button'),
  ).toHaveCount(6);
  await page
    .getByRole('list', { name: 'Etapas do procedimento adquirido' })
    .getByRole('button')
    .first()
    .click();
  await expect(page.getByRole('dialog').locator('pre')).toContainText('identity');
  await page.getByRole('button', { name: 'Fechar', exact: true }).click();
  const artifact = await request.get(`/api/artifact?id=${state.id}&name=goal-1.axgoal`);
  expect(artifact.ok()).toBe(true);
  expect((await artifact.body()).subarray(0, 8).toString()).toBe('AXGP0001');
  const log = (await (await request.get(`/api/export?id=${state.id}`)).text())
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  expect(log.some((e) => e.event === 'controller_drift')).toBe(true);
  expect(log.some((e) => e.event === 'controller_repair')).toBe(true);
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'goal-controller-desktop.png'), fullPage: true });
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark')
      await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    await settledTheme(page);
    const result = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(
      result.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })),
    ).toEqual([]);
  }
  await page.setViewportSize({ width: 320, height: 900 });
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'goal-controller-mobile.png'), fullPage: true });
});

test('common genome: actual births, 256 candidates, inherited procedures and decimal download', async ({
  page,
  request,
}) => {
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('genome');
  await page.getByLabel('Validacoes por caso', { exact: true }).fill('100');
  await expect(page.getByLabel('Seed aleatoria', { exact: true })).toHaveValue('84371921');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(
      async () => {
        const job = (await (await request.get('/api/state')).json()).current;
        return `${job.mode}:${job.status}`;
      },
      { timeout: 60000 },
    )
    .toBe('genome:complete');
  const state = (await (await request.get('/api/state')).json()).current;
  expect(state.candidates).toHaveLength(256);
  expect(state.result.attempts).toBe(1536);
  expect(state.trace.gaps).toBe(0);
  expect(state.cases.some((c: RecordEvent) => c.steps?.length === 6)).toBe(true);
  const candidate = state.candidates.find((c: RecordEvent) => c.frontier && c.artifact);
  const number = await (
    await request.get(`/api/artifact?id=${state.id}&name=${candidate.artifact}`)
  ).text();
  expect(number).toMatch(/^1\d+$/);
  expect(number.length).toBeGreaterThan(10000);
  await page.getByLabel('Buscar artefatos', { exact: true }).fill('founder.axnum');
  await expect(
    page.getByRole('region', { name: 'Artefatos exportaveis' }).getByRole('link'),
  ).toHaveCount(1);
  await page.getByLabel('Buscar artefatos', { exact: true }).fill('');
  const filesBounds = await page
    .getByRole('region', { name: 'Artefatos exportaveis' })
    .boundingBox();
  expect(filesBounds!.height).toBeLessThanOrEqual(420);
  const bounds = await page
    .getByRole('region', { name: 'Populacao de seeds', exact: true })
    .boundingBox();
  expect(bounds!.height).toBeLessThanOrEqual(440);
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'common-genome-desktop.png'), fullPage: true });
  const audit = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(audit.violations.map((v) => v.id)).toEqual([]);
  await page.setViewportSize({ width: 320, height: 900 });
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'common-genome-mobile.png'), fullPage: true });
});

test('language: sourced dictionary, learned numeric answers, domain refusals and real artifacts', async ({
  page,
  request,
}) => {
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('language');
  await page.getByLabel('Validacoes por caso', { exact: true }).fill('100');
  await expect(page.getByLabel('Seed aleatoria', { exact: true })).toHaveValue('71582347');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(
      async () => {
        const job = (await (await request.get('/api/state')).json()).current;
        return `${job.mode}:${job.status}`;
      },
      { timeout: 60000 },
    )
    .toBe('language:complete');
  const job = (await (await request.get('/api/state')).json()).current;
  expect(job.result.correct).toBe(388);
  expect(job.result.refused).toBe(12);
  expect(job.result.wrong).toBe(0);
  expect(job.result.negative_pass).toBe(4);
  expect(job.trace.gaps).toBe(0);
  expect(job.cases).toHaveLength(4);
  const capsule = await request.get(`/api/artifact?id=${job.id}&name=math-2.axnum`);
  expect(capsule.ok()).toBe(true);
  expect((await capsule.body()).subarray(0, 8).toString()).toBe('AXNP0001');
  await expect(page.locator('.lab-metrics').first()).toContainText('RECUSAS DE DOMINIO');
  const outputs = await request.get(`/api/artifact?id=${job.id}&name=predictions.tsv`);
  expect(outputs.ok()).toBe(true);
  const lines = (await outputs.text()).trim().split('\n');
  expect(lines).toHaveLength(401);
  expect(lines.filter((l) => l.includes('\tdomain-refusal\t'))).toHaveLength(12);
  expect(lines.filter((l) => l.includes('\twrong\t'))).toHaveLength(0);
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'grounded-language-desktop.png'), fullPage: true });
  const audit = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(audit.violations.map((v) => v.id)).toEqual([]);
  await page.setViewportSize({ width: 320, height: 900 });
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'grounded-language-mobile.png'), fullPage: true });
});

test('typed task form, live entry-exit output, modal focus stability and corrupt preferences', async ({
  page,
  request,
}) => {
  await page.addInitScript(() => localStorage.setItem('axon.panels', '{"invalid":true}'));
  await ready(page);
  await page.getByRole('button', { name: 'Nova execucao', exact: true }).first().click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('math');
  const input = page.getByRole('textbox', { name: 'Tarefa AXON', exact: true });
  await input.focus();
  // Wait for a real parent clock render, not for a hypothetical animation delay.
  await page.waitForTimeout(1200);
  await expect(input).toBeFocused();
  await page.getByRole('dialog').getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => {
      const s = await (await request.get('/api/state')).json();
      return `${s.current.mode}:${s.current.status}`;
    })
    .toBe('math:complete');
  const state = await (await request.get('/api/state')).json();
  expect(state.current.mode).toBe('math');
  expect(state.current.output.length).toBeGreaterThan(20);
  await page.getByRole('button', { name: 'Execucoes', exact: true }).click();
  await expect(page.locator('.output-code')).toHaveText(state.current.output.trim());
  await page.getByRole('button', { name: 'Nova execucao', exact: true }).first().click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('factor');
  await page.getByRole('dialog').getByLabel('Iniciar pausado').check();
  await page.getByRole('dialog').getByRole('button', { name: 'Executar', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Cancelar execucao', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Cancelar execucao', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.status)
    .toBe('cancelled');
});

test('five-channel composition: high-volume live counters and complete verified graph', async ({
  page,
  request,
}) => {
  await ready(page);
  await page.getByRole('button', { name: 'Nova execucao', exact: true }).first().click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('factor');
  await page.getByRole('slider', { name: 'Canais', exact: true }).fill('5');
  await page.getByRole('dialog').getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.status, {
      timeout: 60000,
    })
    .toBe('complete');
  await expect(page.locator('.three-metrics')).toContainText('Exata');
  const state = await (await request.get('/api/state')).json();
  expect(state.graph.models.length).toBe(5);
  expect(state.current.result.exact).toBe(true);
  expect(state.current.trace.records).toBeGreaterThan(10000);
  expect(state.current.trace.gaps).toBe(0);
  expect(state.current.counters.physical + state.current.counters.cached).toBe(
    state.current.counters.queryCalls,
  );
  await expect(page.locator('.matrix-row')).toHaveCount(5);
  await expect(page.locator('.runtime-row').first()).toContainText(
    new Intl.NumberFormat('pt-BR').format(state.current.trace.records),
  );
  await page.screenshot({ path: path.join(qa, 'real-five-channel-result.png'), fullPage: true });
  await page.getByRole('button', { name: 'Rede de estados', exact: true }).click();
  await pixels(page, '.network-svg');
  await page.screenshot({ path: path.join(qa, 'real-five-channel-network.png'), fullPage: true });
});

test('cumulative: real inherited learning, matched comparisons, file receipts and responsive historical replay', async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('cumulative');
  await page.getByLabel('Validacoes por caso', { exact: true }).fill('100');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.status, {
      timeout: 60000,
    })
    .toBe('complete');
  const state = await (await request.get('/api/state')).json();
  expect(state.current.mode).toBe('cumulative');
  expect(state.current.result.checked).toBe(3200);
  expect(state.current.result.wrong).toBe(0);
  expect(state.current.result.skills).toBe(8);
  expect(state.current.trace.gaps).toBe(0);
  const comparisons = page.getByRole('region', { name: 'Comparacoes de aprendizado', exact: true });
  await expect(comparisons.locator('tbody tr')).toHaveCount(32);
  const plans = page.getByRole('region', { name: 'Planos de arquivos', exact: true });
  await expect(plans.locator('tbody tr')).toHaveCount(3);
  await expect(plans).toContainText('108,00');
  await expect(plans).toContainText('21,00');
  await expect(
    page.getByRole('region', { name: 'Populacao cumulativa', exact: true }).locator('tbody tr'),
  ).toHaveCount(16);
  const recorded = state.current.cases.filter((c: RecordEvent) => c.arm);
  expect(recorded).toHaveLength(32);
  expect(
    recorded.filter((c: RecordEvent) => c.arm === 'founder-random' && c.novel && c.admitted),
  ).toHaveLength(0);
  expect(
    recorded.filter((c: RecordEvent) => c.arm === 'learned-active' && c.novel && c.admitted),
  ).toHaveLength(3);
  const artifact = await request.get(`/api/artifact?id=${state.current.id}&name=plans.csv`);
  expect(artifact.ok()).toBe(true);
  expect(await artifact.text()).toContain('10800');
  const before = JSON.stringify(state.current.samples);
  await page.reload();
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await expect(
    page.locator('.evolution-toolbar').getByText('Historico', { exact: true }),
  ).toBeAttached();
  await expect
    .poll(async () =>
      JSON.stringify((await (await request.get('/api/state')).json()).current.samples),
    )
    .toBe(before);
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark')
      await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    await settledTheme(page);
    await noOverflow(page);
    const report = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(
      report.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })),
    ).toEqual([]);
  }
  await page.screenshot({ path: path.join(qa, 'cumulative-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 320, height: 900 });
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'cumulative-mobile.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('relational: real P0 induction, structural selection, births and honest language limits', async ({
  page,
  request,
}) => {
  test.skip(process.env.AXON_UI_REAL_CORE !== '1', 'Opt-in real-core campaign');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('semantic');
  await page.getByLabel('Validacoes por caso', { exact: true }).fill('100');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current.status, {
      timeout: 60000,
    })
    .toBe('complete');
  const state = await (await request.get('/api/state')).json();
  expect(state.current.mode).toBe('semantic');
  expect(state.current.result.candidates).toBe(768);
  expect(state.current.result.checked).toBe(912);
  expect(state.current.result.natural_correct).toBe(0);
  expect(state.current.result.natural_checked).toBe(20);
  expect(state.current.result.outcome).toContain('partial');
  expect(state.current.trace.gaps).toBe(0);
  expect(state.current.candidates).toHaveLength(768);
  await expect(
    page.getByRole('region', { name: 'Selecao estrutural R1', exact: true }).locator('tbody tr'),
  ).toHaveCount(12);
  await expect(
    page
      .getByRole('region', { name: 'Validacao independente R1', exact: true })
      .locator('tbody tr'),
  ).toHaveCount(8);
  const predictions = page.getByRole('region', { name: 'Respostas observadas R1', exact: true });
  await expect(predictions).toContainText('Ines recebeu 4 livros');
  await expect(predictions).toContainText('Nao resolvido');
  const artifact = await request.get(`/api/artifact?id=${state.current.id}&name=predictions.tsv`);
  expect(artifact.ok()).toBe(true);
  expect(await artifact.text()).toContain('natural-probe');
  await page.screenshot({ path: path.join(qa, 'relational-desktop.png'), fullPage: true });
  await page.reload();
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await expect(
    page.getByRole('region', { name: 'Selecao estrutural R1', exact: true }).locator('tbody tr'),
  ).toHaveCount(12);
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await noOverflow(page);
  }
  await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
  await settledTheme(page);
  const report = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(report.violations.map((v) => v.id)).toEqual([]);
  await page.screenshot({ path: path.join(qa, 'relational-mobile-dark.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('nursery: persistent unique DNA, growth, batch resume and responsive telemetry', async ({
  page,
  request,
}) => {
  test.skip(process.env.AXON_UI_REAL_CORE !== '1', 'Opt-in real-core campaign');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('nursery');
  await page.getByLabel('Novas seeds por lote', { exact: true }).fill('100');
  const genomes = new Set<string>();
  for (const total of [100, 200]) {
    await page.getByRole('button', { name: 'Executar', exact: true }).click();
    await expect
      .poll(async () => (await (await request.get('/api/state')).json()).current.status, {
        timeout: 60000,
      })
      .toBe('complete');
    const state = await (await request.get('/api/state')).json();
    expect(state.current.result.candidates).toBe(total);
    expect(state.current.result.new_candidates).toBe(100);
    expect(state.current.result.admitted).toBe(false);
    expect(state.current.trace.gaps).toBe(0);
    const artifact = await request.get(
      `/api/artifact?id=${state.current.id}&name=new-candidates.tsv`,
    );
    expect(artifact.ok()).toBe(true);
    expect(
      (
        await request.get(`/api/artifact?id=${state.current.id}&name=best-candidate%2Fseed.axnum`)
      ).ok(),
    ).toBe(true);
    expect(
      (await request.get(`/api/artifact?id=${state.current.id}&name=registry-snapshot.axep`)).ok(),
    ).toBe(true);
    const lines = (await artifact.text()).trim().split('\n').slice(1);
    expect(lines).toHaveLength(100);
    for (const row of lines) {
      const dna = row.split('\t')[0];
      expect(genomes.has(dna)).toBe(false);
      genomes.add(dna);
    }
    await expect(
      page.getByRole('heading', { name: 'Viveiro de seeds', exact: true }),
    ).toBeVisible();
  }
  await page.screenshot({ path: path.join(qa, 'nursery-desktop.png'), fullPage: true });
  await page.reload();
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await expect(
    page.getByRole('region', { name: 'Evolucao sem repeticao', exact: true }),
  ).toContainText('200');
  await page.setViewportSize({ width: 320, height: 1000 });
  await noOverflow(page);
  await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
  await settledTheme(page);
  const report = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
    .analyze();
  expect(report.violations.map((v) => v.id)).toEqual([]);
  await page.screenshot({ path: path.join(qa, 'nursery-mobile-dark.png'), fullPage: true });
  expect(errors).toEqual([]);
});

for (const curriculum of ['legacy', 'arithmetic'] as const) {
  test(`constructive: real circuits ${curriculum}, portable seed and responsive evidence`, async ({
    page,
    request,
  }) => {
    test.skip(process.env.AXON_UI_REAL_CORE !== '1', 'Opt-in real-core campaign');
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await ready(page);
    await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
    await page.getByLabel('Experimento', { exact: true }).selectOption('constructive');
    await page
      .getByLabel('Busca construtiva', { exact: true })
      .selectOption(curriculum === 'legacy' ? 'elitist' : 'reinforced');
    await page.getByLabel('Curriculo construtivo', { exact: true }).selectOption(curriculum);
    await expect(page.getByLabel('RNG da busca', { exact: true })).toHaveValue('75218341');
    await page.getByLabel('Novos grafos por tarefa', { exact: true }).fill('100');
    const created = page.waitForResponse(
      (r) => r.url().endsWith('/api/run') && r.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Executar', exact: true }).click();
    expect((await created).ok()).toBe(true);
    await expect
      .poll(async () => (await (await request.get('/api/state')).json()).current?.status, {
        timeout: 60000,
      })
      .toBe('complete');
    const state = await (await request.get('/api/state')).json();
    expect(state.current.mode).toBe('constructive');
    expect(state.current.trace.gaps).toBe(0);
    const tasks = curriculum === 'legacy' ? 6 : 10;
    expect(state.current.result.tasks).toBe(tasks);
    expect(state.current.result.new_candidates).toBeGreaterThan(0);
    expect(state.current.result.new_candidates).toBeLessThanOrEqual(tasks * 100);
    for (const c of state.current.cases) expect(c.correct + c.wrong + c.refused).toBe(1024);
    const number = (n: number) => new Intl.NumberFormat('pt-BR').format(n);
    const rows = page
      .getByRole('region', { name: 'Resultados por caso', exact: true })
      .getByRole('row');
    for (const c of state.current.cases) {
      await expect(
        rows
          .nth(c.case + 1)
          .getByRole('cell')
          .nth(4),
      ).toHaveText(`${number(c.correct)} / ${number(c.checked)}`);
      await expect(
        rows
          .nth(c.case + 1)
          .getByRole('cell')
          .nth(1),
      ).toHaveText(c.admitted ? 'Admitido' : 'Nao admitido');
    }
    const total = state.current.cases.reduce(
      (n: number, c: { correct: number }) => n + c.correct,
      0,
    );
    await expect(page.locator('.lab-metrics').first()).toContainText(
      `${number(total)} / ${number(tasks * 1024)}`,
    );
    const artifact = (name: string) =>
      request.get(`/api/artifact?id=${state.current.id}&name=${encodeURIComponent(name)}`);
    for (const name of [
      'results.tsv',
      'protocol.txt',
      'task-0-circuit.txt',
      'task-0-journal.axep',
      'seed.axnum',
      'birth-result.txt',
    ])
      expect((await artifact(name)).ok()).toBe(true);
    const birth = await (await artifact('birth-result.txt')).text();
    expect(birth).toContain('initial_module_loads=0');
    expect(birth).toContain('active_modules=1');
    expect(birth).toContain('personal_episodes=0');
    await page.screenshot({
      path: path.join(qa, `constructive-${curriculum}-desktop.png`),
      fullPage: true,
    });
    await page.reload();
    await expect(page.getByText('Nucleo conectado', { exact: true })).toBeAttached();
    await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
    await expect(page.getByLabel('Execucao do laboratorio')).toContainText('DNA construtivo');
    await page.setViewportSize({ width: 320, height: 1000 });
    await noOverflow(page);
    await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    await settledTheme(page);
    const report = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(report.violations.map((v) => v.id)).toEqual([]);
    await page.screenshot({
      path: path.join(qa, `constructive-${curriculum}-mobile-dark.png`),
      fullPage: true,
    });
    expect(errors).toEqual([]);
  });
}

test('curriculum: real JSON stages, admission versus transfer, artifacts and responsive results', async ({
  page,
  request,
}) => {
  test.skip(process.env.AXON_UI_REAL_CORE !== '1', 'Opt-in real-core campaign');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('curriculum');
  await expect(page.getByLabel('Curriculo JSON', { exact: true })).toHaveValue('mixed-128-v1');
  await page.getByLabel('Limite de grafos por etapa', { exact: true }).fill('100');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current?.status, {
      timeout: 60000,
    })
    .toBe('complete');
  const state = await (await request.get('/api/state')).json(),
    job = state.current;
  expect(job.mode).toBe('curriculum');
  expect(job.result.tasks).toBe(128);
  expect(job.trace.gaps).toBe(0);
  expect(job.cases).toHaveLength(128);
  expect(job.result.all_goals_passed).toBe(false);
  const rows = page
    .getByRole('region', { name: 'Resultados por caso', exact: true })
    .getByRole('row');
  await expect(rows).toHaveCount(129);
  const number = (n: number) => new Intl.NumberFormat('pt-BR').format(n);
  for (const c of [job.cases[0], job.cases[70], job.cases[127]]) {
    await expect(
      rows
        .nth(c.case + 1)
        .getByRole('cell')
        .nth(4),
    ).toHaveText(`${number(c.correct)} / ${number(c.checked)}`);
    await expect(
      rows
        .nth(c.case + 1)
        .getByRole('cell')
        .nth(5),
    ).toHaveText(`${number(c.transfer_correct)} / ${number(c.transfer_total)}`);
  }
  const artifact = (name: string) =>
    request.get(`/api/artifact?id=${job.id}&name=${encodeURIComponent(name)}`);
  for (const name of [
    'curriculum.json',
    'prepared.json',
    'curriculum-results.json',
    'model-index.json',
    'page-0.axnum',
    'written-product-predictions.json',
  ])
    expect((await artifact(name)).ok()).toBe(true);
  const results = await (await artifact('curriculum-results.json')).json();
  expect(results.stages).toHaveLength(128);
  expect(results.pages[0].initial_loads).toBe(0);
  expect(results.pages[0].post_selection_rechecks).toBeGreaterThan(0);
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'curriculum-desktop.png'), fullPage: true });
  await page.reload();
  await expect(page.getByText('Nucleo conectado', { exact: true })).toBeAttached();
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await expect(page.getByLabel('Execucao do laboratorio')).toContainText('Curriculo JSON');
  await page.setViewportSize({ width: 320, height: 1000 });
  await noOverflow(page);
  await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
  await settledTheme(page);
  expect(
    (
      await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
    ).violations.map((v) => v.id),
  ).toEqual([]);
  await page.screenshot({ path: path.join(qa, 'curriculum-mobile-dark.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('transfer and integration: real acquired roots, policy selection and continuous language', async ({
  page,
  request,
}) => {
  test.skip(process.env.AXON_UI_REAL_CORE !== '1', 'Opt-in real-core campaign');
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
  await page.getByLabel('Experimento', { exact: true }).selectOption('curriculum');
  await page.getByLabel('Curriculo JSON', { exact: true }).fill('transfer-v2');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current?.status, {
      timeout: 60000,
    })
    .toBe('complete');
  let state = await (await request.get('/api/state')).json();
  expect(state.current.result.all_goals_passed).toBe(true);
  expect(state.current.result.tasks).toBe(34);
  expect(state.current.trace.gaps).toBe(0);
  const root = await request.get(`/api/artifact?id=${state.current.id}&name=quadratic-plus.axmath`);
  expect(root.ok()).toBe(true);
  await expect(
    page.getByRole('region', { name: 'Resultados por caso', exact: true }).getByRole('row'),
  ).toHaveCount(35);
  await noOverflow(page);
  await page.screenshot({ path: path.join(qa, 'transfer-v2-desktop.png'), fullPage: true });
  await page.getByLabel('Experimento', { exact: true }).selectOption('integration');
  await page.getByRole('button', { name: 'Executar', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get('/api/state')).json()).current?.status, {
      timeout: 60000,
    })
    .toBe('complete');
  state = await (await request.get('/api/state')).json();
  expect(state.current.result.all_goals_passed).toBe(true);
  expect(state.current.trace.gaps).toBe(0);
  const result = await (
    await request.get(`/api/artifact?id=${state.current.id}&name=audit-integration-v1.json`)
  ).json();
  expect(result.language.updates).toBe(1);
  expect(result.language.old_binding_retained).toBe(true);
  expect(result.meta.cold_succeeded).toEqual([false, false]);
  expect(result.meta.trials.some((t: { admitted: boolean }) => !t.admitted)).toBe(true);
  expect(result.continuous.correct).toBeLessThan(result.continuous.scored);
  await page.setViewportSize({ width: 320, height: 1000 });
  await noOverflow(page);
  await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
  await settledTheme(page);
  expect(
    (
      await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
    ).violations.map((v) => v.id),
  ).toEqual([]);
  await page.screenshot({ path: path.join(qa, 'integration-mobile-dark.png'), fullPage: true });
  expect(errors).toEqual([]);
});

for (const scenario of [
  {
    id: 'structural-v3',
    cases: 5100,
    turns: 2100,
    restarts: 300,
    version: 2,
    families: 9,
    programs: 0,
  },
  {
    id: 'compositional-v1',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 3,
    families: 13,
    programs: 8,
  },
  {
    id: 'agent-v1',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 3,
    families: 13,
    programs: 24,
  },
  {
    id: 'algorithms-v3',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 5,
    families: 13,
    programs: 8,
  },
  {
    id: 'abstractions-v2',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 5,
    families: 13,
    programs: 8,
  },
  {
    id: 'stateful-v1',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 6,
    families: 13,
    programs: 8,
  },
  {
    id: 'inquiry-v1',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 6,
    families: 13,
    programs: 8,
  },
  {
    id: 'm0-state-full-v3',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 6,
    families: 13,
    programs: 8,
  },
  {
    id: 'm1-composition-v1',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 6,
    families: 13,
    programs: 10,
  },
  {
    id: 'm1-context-v1',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 6,
    families: 13,
    programs: 10,
  },
  {
    id: 'm1-lexical-v2',
    cases: 7400,
    turns: 3100,
    restarts: 500,
    version: 7,
    families: 13,
    programs: 10,
  },
])
  test(`structural learning ${scenario.id}: real calculus, dialogue, online teaching and portable rules`, async ({
    page,
    request,
  }) => {
    test.skip(process.env.AXON_UI_REAL_CORE !== '1', 'Opt-in real-core campaign');
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await ready(page);
    await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
    await page.getByLabel('Experimento', { exact: true }).selectOption('structural');
    await expect(page.getByLabel('Curriculo JSON', { exact: true })).toHaveValue(
      'compositional-v1',
    );
    await page.getByLabel('Curriculo JSON', { exact: true }).fill(scenario.id);
    expect(
      await page
        .getByLabel('Curriculo JSON', { exact: true })
        .evaluate((input: HTMLInputElement) => input.checkValidity()),
    ).toBe(true);
    await expect(page.getByText('Seed aleatoria', { exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Executar', exact: true }).click();
    await expect
      .poll(async () => (await (await request.get('/api/state')).json()).current?.status, {
        timeout: ['m1-context-v1', 'm1-lexical-v2'].includes(scenario.id) ? 120000 : 60000,
      })
      .toBe('complete');
    const state = await (await request.get('/api/state')).json();
    expect(state.current.result.admitted).toBe(true);
    expect(state.current.result.correct).toBe(scenario.cases);
    expect(state.current.result.dialogue_correct).toBe(scenario.turns);
    expect(state.current.result.derived_programs).toBe(scenario.programs);
    expect(state.current.trace.gaps).toBe(0);
    if (scenario.id === 'algorithms-v3') {
      expect(state.current.result.algorithm_correct).toBe(1714);
      expect(state.current.result.algorithm_total).toBe(1714);
      expect(state.current.result.total_rules).toBe(67);
      const controls = page.getByRole('region', { name: 'Controles dos algoritmos', exact: true });
      await expect(controls.locator('tbody tr')).toHaveCount(3);
      const result = await (
        await request.get(
          `/api/artifact?id=${state.current.id}&name=algorithm-teaching-result.json`,
        )
      ).json();
      expect(result.controls.map((v: { acquired: number }) => v.acquired)).toEqual([16, 14, 16]);
      const file = await (
        await request.get(`/api/artifact?id=${state.current.id}&name=algorithm-file-result.json`)
      ).json();
      expect(file.actual).toBe(15074);
      const drift = await (
        await request.get(
          `/api/artifact?id=${state.current.id}&name=algorithm-change-0-result.json`,
        )
      ).json();
      expect(drift.suspended).toBe(true);
      expect(drift.relearned).toBe(true);
      const records = (await (await request.get(`/api/export?id=${state.current.id}`)).text())
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      expect(
        records.some(
          (r: { event: string; phase: string }) =>
            r.event === 'algorithm_investigation' && r.phase === 'consolidation',
        ),
      ).toBe(true);
      expect(records.some((r: { event: string }) => r.event === 'algorithm_drift')).toBe(true);
    }
    if (scenario.id === 'abstractions-v2') {
      expect(state.current.result.algorithm_correct).toBe(2360);
      expect(state.current.result.algorithm_total).toBe(2360);
      expect(state.current.result.total_rules).toBe(76);
      const result = await (
        await request.get(
          `/api/artifact?id=${state.current.id}&name=algorithm-teaching-result.json`,
        )
      ).json();
      expect(result.controls.map((v: { acquired: number }) => v.acquired)).toEqual([25, 19, 25]);
      const records = (await (await request.get(`/api/export?id=${state.current.id}`)).text())
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      expect(
        records.some(
          (r: RecordEvent) =>
            r.event === 'algorithm_investigation' &&
            (r.templates ?? 0) > 0 &&
            (r.mining_work ?? 0) > 0,
        ),
      ).toBe(true);
      expect(
        records.some((r: RecordEvent) => r.event === 'algorithm_drift' && (r.invalidated ?? 0) > 1),
      ).toBe(true);
      await expect(
        page.getByRole('region', { name: 'Ultima busca de abstracoes', exact: true }),
      ).toBeVisible();
    }
    if (scenario.id === 'stateful-v1') {
      expect(state.current.result.algorithm_correct).toBe(3365);
      expect(state.current.result.algorithm_total).toBe(3365);
      expect(state.current.result.total_rules).toBe(86);
      const result = await (
        await request.get(
          `/api/artifact?id=${state.current.id}&name=algorithm-teaching-result.json`,
        )
      ).json();
      expect(result.controls.map((v: { acquired: number }) => v.acquired)).toEqual([35, 23, 35]);
      const records = (await (await request.get(`/api/export?id=${state.current.id}`)).text())
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      expect(
        records.some(
          (r: RecordEvent) =>
            r.event === 'algorithm_investigation' &&
            r.operation === 'histogram' &&
            r.library_programs === 32,
        ),
      ).toBe(true);
      const region = page.getByRole('region', { name: 'Ultima busca de abstracoes', exact: true });
      await expect(region).toContainText('BIBLIOTECA / ULTIMA INVESTIGACAO');
    }
    if (scenario.id === 'inquiry-v1') {
      expect(state.current.result.algorithm_correct).toBe(3865);
      expect(state.current.result.algorithm_total).toBe(3865);
      expect(state.current.result.total_rules).toBe(91);
      expect(state.current.algorithmInquiry.phase).toBe('query_observed');
      const records: RecordEvent[] = (
        await (await request.get(`/api/export?id=${state.current.id}`)).text()
      )
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      const selected = records.filter(
        (r) => r.event === 'algorithm_investigation' && r.phase === 'query_selected',
      );
      expect(selected.length).toBeGreaterThan(0);
      expect(selected.every((r) => JSON.parse(r.inquiry!).actual === null)).toBe(true);
      expect(
        records.some((r) => r.phase === 'query_observed' && (r.query_hypotheses ?? 0) > 1),
      ).toBe(true);
      await expect(
        page
          .getByRole('region', { name: 'Controles dos algoritmos', exact: true })
          .locator('tbody tr'),
      ).toHaveCount(4);
      await page
        .getByRole('button', { name: 'Inspecionar consulta de aprendizagem', exact: true })
        .click();
      await expect(page.getByRole('dialog').locator('pre')).toContainText('actual');
      await expect(page.getByRole('dialog').locator('pre')).toContainText('query_observed');
      await page.getByRole('button', { name: 'Fechar', exact: true }).click();
    }
    if (scenario.id === 'm0-state-full-v3') {
      expect(state.current.result.algorithm_correct).toBe(4693);
      expect(state.current.result.algorithm_total).toBe(4693);
      expect(state.current.result.total_rules).toBe(107);
      const region = page.getByRole('region', { name: 'Ultima busca de abstracoes', exact: true });
      await expect(region).toContainText('JANELAS / BUSCA FINAL');
      const records: RecordEvent[] = (
        await (await request.get(`/api/export?id=${state.current.id}`)).text()
      )
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      expect(
        records.some((r) => r.event === 'algorithm_investigation' && (r.library_windows ?? 0) > 1),
      ).toBe(true);
      await expect(
        page.getByRole('button', { name: 'Inspecionar consulta de aprendizagem', exact: true }),
      ).toBeVisible();
    }
    if (['m1-context-v1', 'm1-lexical-v2'].includes(scenario.id)) {
      expect(state.current.result.session_correct).toBe(1536);
      expect(state.current.result.session_total).toBe(1536);
      expect(state.current.result.algorithm_correct).toBe(
        scenario.id === 'm1-lexical-v2' ? 4821 : 4693,
      );
      expect(state.current.result.agent_correct).toBe(1088);
      expect(state.current.result.total_rules).toBe(126);
      expect(
        state.current.cases.filter((c: RecordEvent) => c.method === 'Session::turn'),
      ).toHaveLength(3);
      expect(
        state.current.cases.find(
          (c: RecordEvent) => c.method === 'Session::ground' && c.variant === 'context',
        ).grounded,
      ).toBe(0);
      expect(
        JSON.parse(
          state.current.cases.find(
            (c: RecordEvent) => c.method === 'Session::turn' && c.variant === 'context',
          ).actual_json,
        ).status,
      ).toBe('unit-mismatch');
      await expect(
        page.getByRole('region', { name: 'Sessao contextual', exact: true }),
      ).toContainText('1.536 / 1.536');
      await expect(
        page
          .getByRole('region', { name: 'Controles da sessao contextual', exact: true })
          .locator('tbody tr'),
      ).toHaveCount(3);
      const predictions = await (
        await request.get(`/api/artifact?id=${state.current.id}&name=session-predictions.json`)
      ).json();
      expect(
        predictions
          .filter((r: { variant: string }) => r.variant === 'context')
          .every((r: { correct: boolean }) => r.correct),
      ).toBe(true);
      expect(
        predictions
          .filter((r: { actual?: { status: string } }) => r.actual?.status === 'unit-mismatch')
          .every(
            (r: { queries: number; episodes_before: number; episodes_after: number }) =>
              r.queries === 0 && r.episodes_before === r.episodes_after,
          ),
      ).toBe(true);
      const records: RecordEvent[] = (
        await (await request.get(`/api/export?id=${state.current.id}`)).text()
      )
        .trim()
        .split('\n')
        .map((line: string) => JSON.parse(line));
      const pending = new Map<string, number>();
      let decisions = 0;
      for (const r of records) {
        const key = `${r.variant}/${r.stage}/${r.turn}`;
        if (r.event === 'session_interpretation') {
          expect(r.actual_json).toBeNull();
          pending.set(key, r.coreSeq!);
          decisions++;
        }
        if (r.event === 'session_turn') {
          expect(pending.get(key)).toBeLessThan(r.coreSeq!);
          pending.delete(key);
        }
      }
      expect(decisions).toBe(4608);
      expect(pending.size).toBe(0);
      await page.reload();
      await page.getByRole('button', { name: 'Laboratorio', exact: true }).click();
      await page
        .getByRole('button', { name: 'Inspecionar sessao contextual', exact: true })
        .click();
      await expect(page.getByRole('dialog').locator('pre')).toContainText('unit-mismatch');
      await page.getByRole('button', { name: 'Fechar', exact: true }).click();
    }
    if (scenario.id === 'm1-lexical-v2') {
      expect(state.current.result.file_correct).toBe(9);
      expect(state.current.result.file_total).toBe(9);
      const region = page.getByRole('region', {
        name: 'Arquivos com esquema aprendido',
        exact: true,
      });
      await expect(region).toContainText('9 / 9');
      await page
        .getByRole('button', { name: 'Inspecionar arquivos com esquema aprendido', exact: true })
        .click();
      await expect(page.getByRole('dialog').locator('pre')).toContainText('file_plan');
      await expect(page.getByRole('dialog').locator('pre')).toContainText('clarify-schema');
      await page.getByRole('button', { name: 'Fechar', exact: true }).click();
      const files = await (
        await request.get(`/api/artifact?id=${state.current.id}&name=file-predictions.json`)
      ).json();
      expect(files).toHaveLength(9);
      expect(files.every((r: { correct: boolean }) => r.correct)).toBe(true);
      expect(
        files.every(
          (r: { actual: { teacher_queries: number; seed_unchanged: boolean } }) =>
            r.actual.teacher_queries === 0 && r.actual.seed_unchanged,
        ),
      ).toBe(true);
    }
    if (scenario.id === 'm1-composition-v1') {
      expect(state.current.result.agent_correct).toBe(1088);
      expect(state.current.result.agent_portable).toBe(1088);
      expect(state.current.result.algorithm_correct).toBe(4693);
      expect(state.current.result.total_rules).toBe(115);
      const controls = page.getByRole('region', { name: 'Controles da investigacao', exact: true });
      await expect(controls).toContainText('Composicao concluida');
      const records: RecordEvent[] = (
        await (await request.get(`/api/export?id=${state.current.id}`)).text()
      )
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      const plans = records.filter(
        (r) => r.event === 'investigation' && r.phase === 'interpretation',
      );
      expect(plans.length).toBeGreaterThan(0);
      expect(
        plans.every(
          (r) => r.actual == null && (r.work ?? 0) > 0 && r.expression?.includes('invoke'),
        ),
      ).toBe(true);
      const summary = records.find((r) => r.method === 'Agent::campaign' && r.variant === 'active');
      expect(summary?.language_work).toBe(1085696);
      const goals = await (
        await request.get(`/api/artifact?id=${state.current.id}&name=agent-predictions.json`)
      ).json();
      expect(
        goals.every(
          (r: { correct: boolean; child_correct: boolean }) => r.correct && r.child_correct,
        ),
      ).toBe(true);
      expect(
        goals.some((r: { error: string }) => r.error === 'conflicting structural interpretations'),
      ).toBe(true);
    }
    if (scenario.id === 'agent-v1') {
      expect(state.current.algorithmInquiry).toBeUndefined();
      expect(state.current.result.agent_correct).toBe(1488);
      expect(state.current.result.agent_portable).toBe(1488);
      expect(state.current.result.total_rules).toBe(79);
      const controls = page.getByRole('region', { name: 'Controles da investigacao', exact: true });
      await expect(controls.locator('tbody tr')).toHaveCount(3);
      await expect(controls.getByRole('row').filter({ hasText: 'Ativa + heranca' })).toContainText(
        '338',
      );
      const records = (await (await request.get(`/api/export?id=${state.current.id}`)).text())
        .trim()
        .split('\n')
        .map((s: string) => JSON.parse(s));
      expect(
        records.some(
          (r: { event: string; phase: string }) =>
            r.event === 'investigation' && r.phase === 'refutation',
        ),
      ).toBe(true);
      expect(
        records.some(
          (r: { event: string; phase: string }) =>
            r.event === 'investigation' && r.phase === 'consolidation',
        ),
      ).toBe(true);
      const result = await (
        await request.get(`/api/artifact?id=${state.current.id}&name=agent-result.json`)
      ).json();
      expect(result.variants.map((v: { correct: number }) => v.correct)).toEqual([
        1488, 1488, 1488,
      ]);
    }
    const id = state.current.id;
    const model = await request.get(`/api/artifact?id=${id}&name=structural.axstruct`);
    expect(model.ok()).toBe(true);
    expect((await model.json()).version).toBe(scenario.version);
    const discovery = await (
      await request.get(`/api/artifact?id=${id}&name=discovery-result.json`)
    ).json();
    expect(discovery.derived_programs).toBe(scenario.programs);
    if (scenario.programs) expect(JSON.stringify(discovery.rules)).toContain('sqrt');
    const turns = await (
      await request.get(`/api/artifact?id=${id}&name=dialogue-predictions.json`)
    ).json();
    expect(turns.some((t: { actual: { status: string } }) => t.actual.status === 'clarify')).toBe(
      true,
    );
    expect(turns.filter((t: { restarted: boolean }) => t.restarted)).toHaveLength(
      scenario.restarts,
    );
    const families = page.getByRole('region', { name: 'Resultados por familia', exact: true });
    await expect(families.locator('tbody tr')).toHaveCount(scenario.families);
    await expect(
      families.getByRole('row').filter({ hasText: 'symbolic-derivative' }),
    ).toContainText('1.200');
    await families.getByRole('button', { name: 'symbolic-derivative', exact: true }).click();
    await expect(page.getByRole('dialog').locator('pre')).toContainText('StructuralSeed::transfer');
    await page.getByRole('button', { name: 'Fechar', exact: true }).click();
    await families.evaluate((element) => {
      element.scrollTop = 0;
    });
    await page.evaluate(() => window.scrollTo(0, 0));
    await noOverflow(page);
    await page.screenshot({ path: path.join(qa, `${scenario.id}-desktop.png`), fullPage: true });
    await page.setViewportSize({ width: 320, height: 1000 });
    await noOverflow(page);
    await page.getByRole('button', { name: 'Tema escuro', exact: true }).click();
    await settledTheme(page);
    expect(
      (
        await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
      ).violations.map((v) => v.id),
    ).toEqual([]);
    await page.screenshot({
      path: path.join(qa, `${scenario.id}-mobile-dark.png`),
      fullPage: true,
    });
    expect(errors).toEqual([]);
  });

test('connection recovery restores authoritative counters and reports missed live delivery', async ({
  page,
  request,
}) => {
  const root = fileURLToPath(new URL('../../', import.meta.url));
  const storageRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'axon-ui-recovery-'));
  const app = await createObservatory({
    root,
    storageRoot,
    port: 0,
    autoSnapshot: false,
  });
  const streams = new Set<ServerResponse>();
  app.server.on('request', (req, res) => {
    if (req.url === '/api/events') {
      streams.add(res);
      res.once('close', () => streams.delete(res));
    }
  });
  try {
    await page.goto(app.url);
    await expect(page.getByText('Nucleo conectado', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Nova execucao', exact: true }).first().click();
    await page.getByRole('dialog').getByLabel('Iniciar pausado').check();
    await page.getByRole('dialog').getByRole('button', { name: 'Executar', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Retomar', exact: true })).toBeEnabled();
    const id = app.state().current!.id;
    expect(await fs.stat(path.join(storageRoot, 'events', `${id}.jsonl`))).toBeTruthy();
    await expect(
      fs.stat(path.join(root, '.axon-memory/observatory/events', `${id}.jsonl`)),
    ).rejects.toMatchObject({ code: 'ENOENT' });
    // Close the actual SSE socket while keeping the Rust process and HTTP API alive.
    for (const stream of streams) stream.destroy();
    await expect(page.getByText('Sem conexao', { exact: true })).toBeVisible();
    const state = await (await request.get(`${app.url}/api/state`)).json();
    const response = await request.post(`${app.url}/api/control`, {
      headers: { 'X-Axon-Session': state.token },
      data: { command: 'step' },
    });
    expect(response.ok()).toBe(true);
    await expect
      .poll(
        async () =>
          (await (await request.get(`${app.url}/api/state`)).json()).current.counters.queryCalls,
      )
      .toBe(1);
    await expect(page.getByText('Nucleo conectado', { exact: true })).toBeVisible();
    await expect(page.getByText('1 NA ENTREGA', { exact: true })).toBeVisible();
    const recovered = await (await request.get(`${app.url}/api/state`)).json();
    await expect(page.locator('.runtime-row').first()).toContainText(
      new Intl.NumberFormat('pt-BR').format(recovered.current.trace.records),
    );
    expect(recovered.current.trace.gaps).toBe(0);
    await page.getByRole('button', { name: 'Cancelar execucao', exact: true }).click();
    await expect
      .poll(async () => (await (await request.get(`${app.url}/api/state`)).json()).current.status)
      .toBe('cancelled');
  } finally {
    await app.close();
    const resolved = await fs.realpath(storageRoot);
    const relative = path.relative(await fs.realpath(os.tmpdir()), resolved);
    expect(relative.startsWith('..') || path.isAbsolute(relative)).toBe(false);
    expect(path.basename(resolved).startsWith('axon-ui-recovery-')).toBe(true);
    await fs.rm(resolved, { recursive: true, force: true });
  }
});
