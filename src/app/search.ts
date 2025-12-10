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

interface SearchFunction<TQuery, TScores extends Record<string, number>> {
  searchEmails(
    query: TQuery,
    emails: Email[]
  ): Promise<WithEmailAndScores<TScores>[]>;
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

type InferScoresType<T extends AnySearchFunction> =
  T extends SearchFunction<any, infer TScores> ? TScores : never;

export async function searchUsingFunction<
  TFunction extends SearchFunction<TQuery, TScores>,
  TQuery,
  TScores extends Record<string, number> = InferScoresType<TFunction>,
>(props: {
  func: TFunction;
  scorer: keyof TScores;
  query: TQuery;
  emails: Email[];
}): Promise<(WithEmailAndScores<TScores> & { score: number })[]> {
  const scores = await props.func.searchEmails(props.query, props.emails);
  const res = scores.map((x) => ({ ...x, score: x.scores[props.scorer] }));

  res.sort((a, b) => b.score - a.score);

  return res;
}

type AnySearchFunction = SearchFunction<any, Record<string, number>>;
type MultipleFunctions = {
  [index: string]: SearchFunction<any, Record<string, number>>;
};
type InferMultipleQuery<TFunctions extends MultipleFunctions> = {
  [index in keyof TFunctions]: InferQuery<TFunctions[index]>;
};
type InferMultipleScores<TFunctions extends MultipleFunctions> =
  UnionToIntersection<InferScores<TFunctions[keyof TFunctions]>> &
    Record<string, number>;

type InferQuery<TFunction extends AnySearchFunction> =
  TFunction extends SearchFunction<infer Query, any> ? Query : never;
type InferScores<TFunction extends AnySearchFunction> =
  TFunction extends SearchFunction<
    any,
    infer Scores extends Record<string, number>
  >
    ? Scores
    : never;

type UnionToIntersection<U> = (U extends any ? (x: U) => void : never) extends (
  x: infer I
) => void
  ? I
  : never;

export class MultiQuerySearchFunction<
  TFunctions extends MultipleFunctions,
> implements SearchFunction<
  InferMultipleQuery<TFunctions>,
  InferMultipleScores<TFunctions>
> {
  private functions: TFunctions;
  constructor(functions: TFunctions) {
    this.functions = functions;
  }
  async searchEmails(
    query: InferMultipleQuery<TFunctions>,
    emails: Email[]
  ): Promise<
    WithEmailAndScores<
      UnionToIntersection<InferScores<TFunctions[keyof TFunctions]>> &
        FusionScore
    >[]
  > {
    const promises = Object.entries(this.functions).map(async (entry) => {
      const [functionName, func] = entry as [
        keyof TFunctions,
        TFunctions[keyof TFunctions],
      ];
      const queryPart = query[functionName];
      const res = await func.searchEmails(queryPart, emails);
      return res;
    });
    const results = await Promise.all(promises);

    const merged = mergeScoreFunctions(...results);

    const fusionScores = reciprocalRankFusion(merged);
    const finalScores = mergeScoreFunction(merged, fusionScores);

    return finalScores as any;
  }
}

export class QuerySelectorSearchFunction<
  TFunction extends SearchFunction<TQuery, TScores>,
  TTotalQuery extends { [index in TName]: TQuery },
  TName extends string,
  TQuery,
  TScores extends Record<string, number>,
> implements SearchFunction<TTotalQuery, TScores> {
  private queryPartName: TName;
  private next: TFunction;

  constructor(queryPartName: TName, next: TFunction) {
    this.queryPartName = queryPartName;
    this.next = next;
  }

  searchEmails(
    query: TTotalQuery,
    emails: Email[]
  ): Promise<{ scores: TScores; email: Email }[]> {
    return this.next.searchEmails(query[this.queryPartName], emails);
  }
}

export class CombinedSearchFunction implements SearchFunction<
  string,
  Bm25Score & EmbeddingScore & FusionScore
> {
  private bm25Function = new Bm25SearchFunction();
  private embeddingFunction = new EmbeddingSearchFunction();

  async searchEmails(
    query: string,
    emails: Email[]
  ): Promise<WithEmailAndScores<Bm25Score & EmbeddingScore & FusionScore>[]> {
    const [bm25, embed] = await Promise.all([
      this.bm25Function.searchEmails(query.split(" "), emails),
      this.embeddingFunction.searchEmails(query, emails),
    ]);

    const scores = mergeScoreFunctions(bm25, embed);
    const fusionResult = reciprocalRankFusion(scores);
    return mergeScoreFunctions(scores, fusionResult);
  }
}

type WithEmailAndScores<S> = { email: Email; scores: S };

// Merge an array of score-lists into one, by email, with precise typing.
// - Accepts any number of lists: [{email, scores: A}][], [{email, scores: B}][], ...
// - Returns [{email, scores: A & B & ...}] with intersection of all score types.
// - Assumes each list has the same emails in the same order.
function mergeScoreFunctions<
  Lists extends ReadonlyArray<
    ReadonlyArray<WithEmailAndScores<Record<string, number>>>
  >,
>(
  ...lists: Lists
): Array<
  Lists extends []
    ? never
    : Lists[number] extends ReadonlyArray<WithEmailAndScores<infer S>>
      ? WithEmailAndScores<UnionToIntersection<S>>
      : never
> {
  if (lists.length === 0) return [];

  const length = lists[0].length;
  // Basic sanity check (optional): ensure all lists have the same length
  // and corresponding emails match.
  // You can remove this block if you don't want runtime checks.
  for (let i = 1; i < lists.length; i++) {
    if (lists[i].length !== length) {
      throw new Error("All score lists must have the same length.");
    }
  }

  const result: any[] = new Array(length);
  for (let i = 0; i < length; i++) {
    const email = lists[0][i].email;
    const merged: Record<string, number> = {};
    for (let l = 0; l < lists.length; l++) {
      const item = lists[l][i];
      // Optional check: ensure emails align
      // if (item.email !== email) throw new Error("Mismatched emails across lists.");
      Object.assign(merged, item.scores);
    }
    result[i] = { email, scores: merged };
  }
  return result as any;
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
  emailScores: WithEmailAndScores<TScores>[]
): WithEmailAndScores<FusionScore>[] {
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

export class Bm25SearchFunction implements SearchFunction<string[], Bm25Score> {
  async searchEmails(
    query: string[],
    emails: Email[]
  ): Promise<WithEmailAndScores<Bm25Score>[]> {
    if (query.length == 0) {
      return emails.map((x) => ({ email: x, scores: { bm25: 0 } }));
    }

    const corpus = emails.map((email) =>
      `${email.subject} ${email.body}`.toLowerCase()
    );

    const scores: number[] = (BM25 as any)(corpus, query);

    const scored = emails.map((x, i) => ({
      scores: {
        bm25: scores[i],
      },
      email: x,
    }));

    return scored;
  }
}

export class EmbeddingSearchFunction implements SearchFunction<
  string,
  EmbeddingScore
> {
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
