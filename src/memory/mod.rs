use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct Episode {
    pub id: u64,
    pub tick: u64,
    pub salience: f32,
    pub trace: String,
    pub recall_score: f32,
}

#[derive(Clone, Debug)]
pub struct MemoryState {
    pub episodes: Vec<Episode>,
    pub short_trace: VecDeque<char>,
    pub next_episode_id: u64,
}

impl MemoryState {
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
            short_trace: VecDeque::with_capacity(256),
            next_episode_id: 1,
        }
    }

    pub fn push_input(&mut self, tick: u64, ch: char, salience: f32) {
        if self.short_trace.len() >= 256 {
            self.short_trace.pop_front();
        }
        self.short_trace.push_back(ch);

        if ch == '\n' || self.short_trace.len() >= 64 {
            let trace: String = self.short_trace.iter().collect();
            let episode = Episode {
                id: self.next_episode_id,
                tick,
                salience,
                trace,
                recall_score: salience,
            };
            self.next_episode_id = self.next_episode_id.saturating_add(1);
            self.episodes.push(episode);
            self.short_trace.clear();
            if self.episodes.len() > 4096 {
                self.episodes.remove(0);
            }
        }
    }

    pub fn decay(&mut self, dt_ticks: u64) {
        if dt_ticks == 0 {
            return;
        }
        let dt = dt_ticks as f32 * 0.01_f32;
        let tau = 15.0_f32;
        let decay_factor = (-dt / tau).exp();
        for episode in &mut self.episodes {
            episode.recall_score *= decay_factor;
            if episode.recall_score < 0.0001 {
                episode.recall_score = 0.0001;
            }
        }
    }

    pub fn reinforce_overlap(&mut self, context: &str) {
        if context.is_empty() {
            return;
        }
        for episode in &mut self.episodes {
            let overlap = overlap_ratio(&episode.trace, context);
            if overlap > 0.0 {
                episode.recall_score = (episode.recall_score + overlap * 0.1).min(1.0);
            }
        }
    }

    pub fn top_recent(&self, n: usize) -> Vec<&Episode> {
        let mut refs: Vec<&Episode> = self.episodes.iter().collect();
        refs.sort_by(|a, b| b.recall_score.total_cmp(&a.recall_score));
        refs.into_iter().take(n).collect()
    }
}

fn overlap_ratio(a: &str, b: &str) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut hits = 0u32;
    let mut total = 0u32;
    for ch in a.chars() {
        total = total.saturating_add(1);
        if b.contains(ch) {
            hits = hits.saturating_add(1);
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f32 / total as f32
    }
}
