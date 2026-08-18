/**
 * Resolves an evidence citation string (an NCT id, PMCID, FDA application
 * number, or SEC accession number) to a real, external URL a reader can
 * click through to. Shared by IndicationMatrix and CatalystTracker --
 * both cite evidence the exact same four ways, since both are grounded in
 * the same underlying Qdrant collections (see research_agent.py).
 *
 * Same accessdata.fda.gov URL shape seed_bulk_data.py already builds
 * server-side for FDA records, and the same sec.gov Archives URL shape
 * fetch_sec_edgar.py's filing_source_url() builds server-side for SEC
 * filings (an accession number's first 10 digits ARE the filer's CIK --
 * verified there directly against a real filing's own SGML header).
 */
export function sourceUrl(source: string): string | null {
  if (/^NCT\d+$/.test(source)) {
    return `https://clinicaltrials.gov/study/${source}`;
  }
  if (/^PMC\d+$/.test(source)) {
    return `https://www.ncbi.nlm.nih.gov/pmc/articles/${source}/`;
  }
  const fdaMatch = /^(BLA|NDA|ANDA)(\d+)$/.exec(source);
  if (fdaMatch) {
    return `https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo=${fdaMatch[2]}`;
  }
  const secMatch = /^(\d{10})-(\d{2})-(\d{6})$/.exec(source);
  if (secMatch) {
    const cik = String(Number(secMatch[1]));
    const accessionNoDashes = source.replace(/-/g, "");
    return `https://www.sec.gov/Archives/edgar/data/${cik}/${accessionNoDashes}/`;
  }
  return null;
}
