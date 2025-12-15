"use client";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  MailIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  Type,
  Brain,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { EmailChunk } from "../search";

function EmailCard({ scoredEmail }: { scoredEmail: ScoredEmail }) {
  const { email, scores } = scoredEmail;
  const [expanded, setExpanded] = useState(false);

  const bm25Score = scores["bm25"]?.toFixed(1) || "N/A";
  const embeddingScore =
    "embedding" in scores ? (scores["embedding"] * 100).toFixed(1) : "N/A";
  const rrfScore =
    "fusion" in scores ? (scores["fusion"] * 100).toFixed(1) : "N/A";

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(date);
  };

  return (
    <Card className="p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 p-1.5 rounded-full bg-primary/10">
          <MailIcon className="h-4 w-4 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4 mb-1">
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-base mb-0.5">
                {email.subject}
              </h3>
              <p className="text-xs text-muted-foreground">{email.from}</p>
            </div>

            <span className="text-xs text-muted-foreground whitespace-nowrap">
              {formatDate(email.timestamp)}
            </span>

            <div className="flex items-center gap-2">
              <span
                className={
                  "text-xs font-mono flex items-center transition-all text-foreground font-medium"
                }
                title="BM25 Score"
              >
                <Type className={"text-blue-400"} />
                {bm25Score}
              </span>
              <span
                className={
                  "text-xs font-mono flex items-center transition-all text-foreground font-medium"
                }
                title="Semantic Score"
              >
                <Brain className={"text-pink-400"} />
                {embeddingScore}%
              </span>
              <span
                className={
                  "text-xs font-mono flex items-center transition-all text-foreground font-medium"
                }
                title="Final RRF Score"
              >
                <Zap
                  className={"w-3 h-3 mr-1.5 transition-all text-yellow-400"}
                />
                {rrfScore}
              </span>
            </div>
          </div>

          <p className="text-sm text-foreground/80 mt-2 line-clamp-2">
            {email.chunk.substring(0, 100) + "..."}
          </p>

          {expanded && (
            <div className="mt-3 pt-3 border-t">
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                  {email.chunk}
                </pre>
              </div>
            </div>
          )}

          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
            className="mt-2 h-8 text-primary hover:text-primary px-2"
          >
            {expanded ? (
              <>
                <ChevronUpIcon className="h-3.5 w-3.5 mr-1" />
                Show less
              </>
            ) : (
              <>
                <ChevronDownIcon className="h-3.5 w-3.5 mr-1" />
                See more
              </>
            )}
          </Button>
        </div>
      </div>
    </Card>
  );
}

type ScoredEmail = {
  email: EmailChunk;
  scores: Record<string, number>;
  score: number;
};

export function EmailList({ scoredEmails }: { scoredEmails: ScoredEmail[] }) {
  if (scoredEmails.length === 0) {
    return (
      <div className="text-center py-12">
        <MailIcon className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No emails found</h3>
        <p className="text-muted-foreground">Try adjusting your search query</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {scoredEmails.map((scoredEmail) => (
        <EmailCard key={`${scoredEmail.email.id}-${scoredEmail.email.index}`} scoredEmail={scoredEmail} />
      ))}
    </div>
  );
}
