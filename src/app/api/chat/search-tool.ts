import {
  loadEmails,
  MultiQuerySearchFunction,
  EmbeddingSearchFunction,
  Bm25SearchFunction,
  searchUsingFunction,
  chunkEmails,
} from "@/app/search";
import { tool } from "ai";
import { z } from "zod";

export const searchTool = tool({
  description:
    "Search emails using at least 5 keywords and a semantic search query. Returns most relevant emails ranked by reciprocal rank fusion. Use synonyms in the keywords, as this does exact matching only!",
  inputSchema: z.object({
    keywords: z
      .array(z.string())
      .describe(
        "Exact keywords for BM25 search (names, amounts, specific terms)"
      )
      .optional(),
    searchQuery: z
      .string()
      .describe("Natural language query for semantic search (broader concepts)")
      .optional(),
  }),
  execute: async ({ keywords, searchQuery }) => {
    console.log("Keywords:", keywords);
    console.log("Search query:", searchQuery);

    const emails = await loadEmails();
    const emailChunks = await chunkEmails(emails);

    const func = new MultiQuerySearchFunction({
      bm25: new Bm25SearchFunction(),
      embedding: new EmbeddingSearchFunction(),
    });

    const values = await searchUsingFunction({
      func,
      scorer: "fusion",
      emailChunks,
      query: {
        bm25: keywords ?? [],
        embedding: searchQuery ?? "",
      },
      includeZeros: false,
      limit: 10,
    });

    console.log("discovered email chunk count: ", values.length);

    return {
      emails: values.map((r) => ({
        id: r.email.id,
        from: r.email.from,
        to: r.email.to,
        subject: r.email.subject,
        body: r.email.chunk,
        timestamp: r.email.timestamp,
        scores: r.scores,
        score: r.score,
      })),
    };
  },
});
