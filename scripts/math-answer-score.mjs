// Separate from the training-expression reader; this never parses calculation traces.
export function finalAnswer(answer){
  if(typeof answer!=='string'||answer.length>16384)throw Error('answer bounds');
  const marker='\n#### ',at=answer.lastIndexOf(marker);if(at<0)return null;
  const final=answer.slice(at+marker.length).trim();
  if(!/^[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$/u.test(final))return null;
  const n=Number(final.replaceAll(',',''));return Number.isFinite(n)?n:null;
}
