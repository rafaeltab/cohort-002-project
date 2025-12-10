import BM25 from "okapibm25";
import fs from "fs/promises";
import path from "path";
import { embedMany, embed, cosineSimilarity } from "ai";
import { google } from "@ai-sdk/google";

export interface Email {
  id: string;
  threadId: string;
  from: string;
  to: string | string[];
  cc?: string[];
  subject: string;
  body: string;
  timestamp: string;
  inReplyTo?: string;
  references?: string[];
  labels?: string[];
  arcId?: string;
  phaseId?: number;
}

interface SearchFunction<TScores> {
  searchEmails(
    query: string,
    emails: Email[]
  ): Promise<{ scores: TScores; email: Email }[]>;
}

type Bm25Score = {
  bm25: number;
};

type EmbeddingScore = {
  embedding: number;
};

type FusionScore = {
  fusion: number;
};

type InferScoresType<T extends SearchFunction<unknown>> =
  T extends SearchFunction<infer TScores> ? TScores : never;

export async function searchUsingFunction<
  TFunction extends SearchFunction<TScores>,
  TScores extends Record<string, number> = InferScoresType<TFunction>,
>(props: {
  func: TFunction;
  scorer: keyof TScores;
  query: string;
  emails: Email[];
}): Promise<{ email: Email; scores: TScores; score: number }[]> {
  const scores = await props.func.searchEmails(props.query, props.emails);
  const res = scores.map((x) => ({ ...x, score: x.scores[props.scorer] }));

  res.sort((a, b) => b.score - a.score);

  return res;
}

export class CombinedSearchFunction implements SearchFunction<
  Bm25Score & EmbeddingScore & FusionScore
> {
  private bm25Function = new Bm25SearchFunction();
  private embeddingFunction = new EmbeddingSearchFunction();

  async searchEmails(
    query: string,
    emails: Email[]
  ): Promise<
    { scores: Bm25Score & EmbeddingScore & FusionScore; email: Email }[]
  > {
    const [bm25, embed] = await Promise.all([
      this.bm25Function.searchEmails(query, emails),
      this.embeddingFunction.searchEmails(query, emails),
    ]);

    const scores = mergeScoreFunction(bm25, embed);
    const fusionResult = reciprocalRankFusion(scores);
    return mergeScoreFunction(scores, fusionResult);
  }
}

function mergeScoreFunction<TAScores, TBScores>(
  scoresA: { email: Email; scores: TAScores }[],
  scoresB: { email: Email; scores: TBScores }[]
): { email: Email; scores: TAScores & TBScores }[] {
  return Array.from(
    zip(scoresA, scoresB, (a, b) => ({
      email: a.email,
      scores: {
        ...a.scores,
        ...b.scores,
      },
    }))
  );
}

function* zip<TA, TB, TC>(
  a: TA[],
  b: TB[],
  combine: (aItem: TA, bItem: TB) => TC
): Generator<TC> {
  for (let i = 0; i < a.length; i++) {
    const aItem = a[i];
    const bItem = b[i];

    yield combine(aItem, bItem);
  }
}

const RRF_K = 60;

function reciprocalRankFusion<TScores extends Record<string, number>>(
  emailScores: { email: Email; scores: TScores }[]
): { email: Email; scores: FusionScore }[] {
  const rankers = Object.keys(emailScores[0].scores);
  const rankings = rankers.map((ranker) =>
    emailScores
      .map((x) => ({ email: x.email, score: x.scores[ranker] }))
      .sort((a, b) => b.score - a.score)
  );

  const rrfScores = new Map<string, number>();

  rankings.forEach((ranking) => {
    ranking.forEach((doc, rank) => {
      const currentScore = rrfScores.get(doc.email.id) || 0;
      const contribution = 1 / (RRF_K + rank);
      rrfScores.set(doc.email.id, currentScore + contribution);
    });
  });

  return emailScores.map((x) => ({
    email: x.email,
    scores: { fusion: rrfScores.get(x.email.id)! },
  }));
}

export class Bm25SearchFunction implements SearchFunction<Bm25Score> {
  async searchEmails(
    query: string,
    emails: Email[]
  ): Promise<{ scores: Bm25Score; email: Email }[]> {
    const keywords = query.split(" ");
    if (keywords.length == 0) {
      return emails.map((x) => ({ email: x, scores: { bm25: 0 } }));
    }

    const corpus = emails.map((email) =>
      `${email.subject} ${email.body}`.toLowerCase()
    );

    const scores: number[] = (BM25 as any)(corpus, keywords);

    const scored = emails.map((x, i) => ({
      scores: {
        bm25: scores[i],
      },
      email: x,
    }));

    return scored;
  }
}

export class EmbeddingSearchFunction implements SearchFunction<EmbeddingScore> {
  async searchEmails(
    query: string,
    emails: Email[]
  ): Promise<{ scores: EmbeddingScore; email: Email }[]> {
    if (query.trim().length == 0) {
      return emails.map((x) => ({ email: x, scores: { embedding: 0 } }));
    }

    const embeddings = await loadOrGenerateEmbeddings(emails);
    const { embedding } = await embed({
      model: google.textEmbeddingModel("text-embedding-004"),
      value: query,
    });

    const scored = embeddings.map((x) => ({
      scores: {
        embedding: cosineSimilarity(x.embedding, embedding),
      },
      email: emails.find((y) => x.id == y.id)!,
    }));

    return scored;
  }
}

export async function loadEmails(): Promise<Email[]> {
  const filePath = path.join(process.cwd(), "data", "emails.json");
  const fileContent = await fs.readFile(filePath, "utf-8");
  return JSON.parse(fileContent);
}

const CACHE_DIR = path.join(process.cwd(), "data", "embeddings");

const CACHE_KEY = "google-text-embedding-004";

const getEmbeddingFilePath = (id: string) =>
  path.join(CACHE_DIR, `${CACHE_KEY}-${id}.json`);

export async function loadOrGenerateEmbeddings(
  emails: Email[]
): Promise<{ id: string; embedding: number[] }[]> {
  // Ensure cache directory exists
  await fs.mkdir(CACHE_DIR, { recursive: true });

  const results: { id: string; embedding: number[] }[] = [];
  const uncachedEmails: Email[] = [];

  // Check cache for each email
  for (const email of emails) {
    try {
      const cached = await fs.readFile(getEmbeddingFilePath(email.id), "utf-8");
      const data = JSON.parse(cached);
      results.push({ id: email.id, embedding: data.embedding });
    } catch {
      // Cache miss - need to generate
      uncachedEmails.push(email);
    }
  }

  // Generate embeddings for uncached emails in batches of 99
  if (uncachedEmails.length > 0) {
    console.log(`Generating embeddings for ${uncachedEmails.length} emails`);

    const BATCH_SIZE = 99;
    for (let i = 0; i < uncachedEmails.length; i += BATCH_SIZE) {
      const batch = uncachedEmails.slice(i, i + BATCH_SIZE);
      console.log(
        `Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(
          uncachedEmails.length / BATCH_SIZE
        )}`
      );

      const { embeddings } = await embedMany({
        model: google.textEmbeddingModel("text-embedding-004"),
        values: batch.map((e) => `${e.subject} ${e.body}`),
      });

      // Write batch to cache
      for (let j = 0; j < batch.length; j++) {
        const email = batch[j];
        const embedding = embeddings[j];

        await fs.writeFile(
          getEmbeddingFilePath(email.id),
          JSON.stringify({ id: email.id, embedding })
        );

        results.push({ id: email.id, embedding });
      }
    }
  }

  return results;
}
