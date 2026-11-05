export function mergeSeedCandidates(candidates=[],event) {
  if(!['seed','common-genome-v1','cumulative-v1','relational-v1'].includes(event.experiment))return candidates;
  if(event.event==='seed_selection'&&Array.isArray(event.selected))return candidates.map(c=>c.generation===event.generation?{...c,frontier:event.selected.includes(c.candidate)}:c);
  const limit=event.experiment==='relational-v1'?768:event.experiment==='common-genome-v1'?256:event.experiment==='cumulative-v1'?16:128;
  if(!['seed_candidate','seed_evaluation'].includes(event.event)||!Number.isInteger(event.candidate)||event.candidate<0||event.candidate>=limit)return candidates;
  const before=candidates.find(c=>c.candidate===event.candidate);
  return [...candidates.filter(c=>c.candidate!==event.candidate),{...before,...event}].sort((a,b)=>a.candidate-b.candidate);
}
export class QueryMemory {
  constructor() { this.nodes = new Map(); this.outputs = new Map(); this.calls = 0; this.hits = 0; }
  clear() { this.nodes.clear(); this.outputs.clear(); this.calls = 0; this.hits = 0; }
  restore(nodes = []) { this.nodes = new Map(nodes.map(node => [node.id, node])); }
  accept(event) {
    if(event.event==='end'){for(const node of this.nodes.values())if(node.root){node.active=false;node.closedBy='process-exit';}return [];}
    if(event.event==='cache_release'){
      const root=this.nodes.get(`cache:${event.context}:root`);if(root)root.active=false;return [];
    }
    const frames=event.frames??(event.event==='step'?[event.reply]:event.reply?.map(output=>[output]))??[];
    if(['query','step'].includes(event.event)&&event.context)for(const frame of frames)frame?.forEach((output,dimension)=>{
      const context=event.frames||event.fromStates?.length>1?`${event.context}/factor-${dimension}`:event.context;
      const id=`output:${context}:${output}`;
      if(!this.outputs.has(id))this.outputs.set(id,{id,context,output,physical:0,cached:0,executed:0});
      const value=this.outputs.get(id);value[event.event==='step'?'executed':event.cacheHit?'cached':'physical']++;
    });
    if (event.event !== 'query' || !Array.isArray(event.actions) || !event.context) return [];
    this.calls++; if (event.cacheHit) this.hits++;
    const root = `cache:${event.context}:root`;
    if (!this.nodes.has(root)) this.nodes.set(root, { id: root, context: event.context, parent: null, depth: 0, root: true, active:true, alphabet:event.actionsCount??16, output: [] });
    const path = []; let parent = root;
    const replies = event.frames ?? event.reply?.map(output => [output]) ?? [];
    for (let i = 0; i < Math.min(event.actions.length, replies.length); i++) {
      const id = `cache:${event.context}:${event.actions.slice(0, i + 1).join('.')}`;
      if (!this.nodes.has(id)) this.nodes.set(id, { id, context: event.context, parent, depth: i + 1, action: event.actions[i], word:event.actions.slice(0,i+1),alphabet:event.actionsCount??16, output: replies[i] });
      path.push(id); parent = id;
    }
    return path;
  }
}
