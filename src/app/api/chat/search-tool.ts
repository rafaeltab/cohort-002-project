import {
  loadEmails,
  MultiQuerySearchFunction,
  EmbeddingSearchFunction,
  Bm25SearchFunction,
  searchUsingFunction,
  chunkEmails,
  RerankerFunction,
} from "@/app/search";
import { tool, UIMessage } from "ai";
import { z } from "zod";

export function searchTool(messages: UIMessage[]) {
  return tool({
    description:
      "Search emails using both keyword and semantic search. Returns metadata with snippets only - use getEmails tool to fetch full content of specific emails.",
    inputSchema: z.object({
      keywords: z
        .array(z.string())
        .describe(
          "Exact keywords for BM25 search (names, amounts, specific terms)"
        )
        .optional(),
      searchQuery: z
        .string()
        .describe(
          "Natural language query for semantic search (broader concepts)"
        )
        .optional(),
    }),
    execute: async ({ keywords, searchQuery }) => {
      console.log("Keywords:", keywords);
      console.log("Search query:", searchQuery);

      const emails = await loadEmails();
      const emailChunks = await chunkEmails(emails);

      const func = new RerankerFunction({
        messages: messages,
        next: new MultiQuerySearchFunction({
          bm25: new Bm25SearchFunction(),
          embedding: new EmbeddingSearchFunction(),
        }),
        chunkCountToReturn: 10,
        orderBy: "fusion",
        queryToString: (q) => `Query: ${q.embedding}
Keywords: ${q.bm25}`,
        chunkCountToRerank: 30,
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

      // Return metadata with snippets only
      const topEmails = values.map((r) => {
        // Get full email to extract threadId
        const fullEmail = emails.find((e) => e.id === r.email.id);
        const snippet =
          r.email.chunk.slice(0, 150).trim() +
          (r.email.chunk.length > 150 ? "..." : "");

        return {
          id: r.email.id,
          threadId: fullEmail?.threadId ?? "",
          subject: r.email.subject,
          from: r.email.from,
          to: r.email.to,
          timestamp: r.email.timestamp,
          score: r.score,
          snippet,
        };
      });

      console.log("Top emails:", topEmails.length);
      return topEmails;
    },
  });
}
