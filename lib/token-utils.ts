import { SessionState, WithParts } from "./state"
import { AssistantMessage, UserMessage } from "@opencode-ai/sdk/v2"
import { Logger } from "./logger"
import * as _anthropicTokenizer from "@anthropic-ai/tokenizer"
const anthropicCountTokens = (_anthropicTokenizer.countTokens ??
    (_anthropicTokenizer as any).default?.countTokens) as typeof _anthropicTokenizer.countTokens
import { getLastUserMessage } from "./messages/query"

export function getCurrentTokenUsage(state: SessionState, messages: WithParts[]): number {
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        if (msg.info.role !== "assistant") {
            continue
        }

        const assistantInfo = msg.info as AssistantMessage
        if ((assistantInfo.tokens?.output || 0) <= 0) {
            continue
        }

        if (
            state.lastCompaction > 0 &&
            (msg.info.time.created < state.lastCompaction ||
                (msg.info.summary === true && msg.info.time.created === state.lastCompaction))
        ) {
            return 0
        }

        const input = assistantInfo.tokens?.input || 0
        const output = assistantInfo.tokens?.output || 0
        const reasoning = assistantInfo.tokens?.reasoning || 0
        const cacheRead = assistantInfo.tokens?.cache?.read || 0
        const cacheWrite = assistantInfo.tokens?.cache?.write || 0

        // [FIX Bug 17] Provider-aware token estimation
        // Anthropic API: input includes cached portion → input >= cacheRead → use input+output+reasoning
        // GLM-5.1 / other providers: input is only non-cached tokens → cacheRead >> input → cacheRead ≈ actual context size
        let contextTokens: number
        if (cacheRead > input && input > 0) {
            // Non-Anthropic provider: cacheRead is the best proxy for current context size
            contextTokens = cacheRead
        } else if (input > 0) {
            // Anthropic-style: input already includes the full context
            contextTokens = input + output + reasoning
        } else {
            // No input tokens reported, use cacheRead as fallback
            contextTokens = cacheRead || 0
        }

        // [DEBUG] Log token breakdown for diagnosis
        try { require("fs").appendFileSync("/tmp/dcp-debug.log", `[getCurrentTokenUsage] ses=${state.sessionId?.slice(-8)} id=${msg.info.id.slice(-8)} tokens={input:${input}, output:${output}, reasoning:${reasoning}, cacheRead:${cacheRead}, cacheWrite:${cacheWrite}} contextTokens=${contextTokens} modelContextLimit=${state.modelContextLimit}\n`) } catch(_e){}
        return contextTokens
    }

    // [FIX Bug 5] fallback: estimate tokens from message content when no assistant
    // message has output tokens (first turn or after full compaction)
    let estimated = 0
    for (const m of messages) {
        const parts = Array.isArray(m.parts) ? m.parts : []
        for (const part of parts) {
            if (part.type === "text" && typeof part.text === "string") {
                estimated += countTokens(part.text)
            }
        }
    }
    // [DEBUG] Log fallback estimation
    try { require("fs").appendFileSync("/tmp/dcp-debug.log", `[getCurrentTokenUsage FALLBACK] ses=${state.sessionId?.slice(-8)} estimated=${estimated} modelContextLimit=${state.modelContextLimit} messages=${messages.length}\n`) } catch(_e){}
    return estimated
}

export function getCurrentParams(
    state: SessionState,
    messages: WithParts[],
    logger: Logger,
): {
    providerId: string | undefined
    modelId: string | undefined
    agent: string | undefined
    variant: string | undefined
} {
    const userMsg = getLastUserMessage(messages)
    if (!userMsg) {
        logger.debug("No user message found when determining current params")
        return {
            providerId: undefined,
            modelId: undefined,
            agent: undefined,
            variant: undefined,
        }
    }
    const userInfo = userMsg.info as UserMessage
    const agent: string = userInfo.agent
    const providerId: string | undefined = userInfo.model.providerID
    const modelId: string | undefined = userInfo.model.modelID
    const variant: string | undefined = userInfo.model.variant

    return { providerId, modelId, agent, variant }
}

export function countTokens(text: string): number {
    if (!text) return 0
    try {
        return anthropicCountTokens(text)
    } catch {
        return Math.round(text.length / 4)
    }
}

export function estimateTokensBatch(texts: string[]): number {
    if (texts.length === 0) return 0
    return countTokens(texts.join(" "))
}

export const COMPACTED_TOOL_OUTPUT_PLACEHOLDER = "[Old tool result content cleared]"

function stringifyToolContent(value: unknown): string {
    return typeof value === "string" ? value : JSON.stringify(value)
}

export function extractCompletedToolOutput(part: any): string | undefined {
    if (
        part?.type !== "tool" ||
        part.state?.status !== "completed" ||
        part.state?.output === undefined
    ) {
        return undefined
    }

    if (part.state?.time?.compacted) {
        return COMPACTED_TOOL_OUTPUT_PLACEHOLDER
    }

    return stringifyToolContent(part.state.output)
}

export function extractToolContent(part: any): string[] {
    const contents: string[] = []

    if (part?.type !== "tool") {
        return contents
    }

    if (part.state?.input !== undefined) {
        contents.push(stringifyToolContent(part.state.input))
    }

    const completedOutput = extractCompletedToolOutput(part)
    if (completedOutput !== undefined) {
        contents.push(completedOutput)
    } else if (part.state?.status === "error" && part.state?.error) {
        contents.push(stringifyToolContent(part.state.error))
    }

    return contents
}

export function countToolTokens(part: any): number {
    const contents = extractToolContent(part)
    return estimateTokensBatch(contents)
}

export function getTotalToolTokens(state: SessionState, toolIds: string[]): number {
    let total = 0
    for (const id of toolIds) {
        const entry = state.toolParameters.get(id)
        total += entry?.tokenCount ?? 0
    }
    return total
}

export function countMessageTextTokens(msg: WithParts): number {
    const texts: string[] = []
    const parts = Array.isArray(msg.parts) ? msg.parts : []
    for (const part of parts) {
        if (part.type === "text") {
            texts.push(part.text)
        }
    }
    if (texts.length === 0) return 0
    return estimateTokensBatch(texts)
}

export function countAllMessageTokens(msg: WithParts): number {
    const parts = Array.isArray(msg.parts) ? msg.parts : []
    const texts: string[] = []
    for (const part of parts) {
        if (part.type === "text") {
            texts.push(part.text)
        } else {
            texts.push(...extractToolContent(part))
        }
    }
    if (texts.length === 0) return 0
    return estimateTokensBatch(texts)
}
