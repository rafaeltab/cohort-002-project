import {
  loadEmails,
  MultiQuerySearchFunction,
  EmbeddingSearchFunction,
  Bm25SearchFunction,
  searchUsingFunction,
} from "@/app/search";
import { tool } from "ai";
import { z } from "zod";

export const searchTool = tool({
  description:
    "Search emails using both keyword and semantic search. Returns most relevant emails ranked by reciprocal rank fusion.",
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

    const func = new MultiQuerySearchFunction({
      bm25: new Bm25SearchFunction(),
      embedding: new EmbeddingSearchFunction(),
    });

    const values = await searchUsingFunction({
      func,
      scorer: "fusion",
      emails,
      query: {
        bm25: keywords ?? [],
        embedding: searchQuery ?? "",
      },
      includeZeros: false,
      limit: 10,
    });

    console.log("discovered emails", values);

    return {
      emails: values.map((r) => ({
        id: r.email.id,
        from: r.email.from,
        to: r.email.to,
        subject: r.email.subject,
        body: r.email.body,
        timestamp: r.email.timestamp,
        scores: r.scores,
        score: r.score,
      })),
    };
  },
});
