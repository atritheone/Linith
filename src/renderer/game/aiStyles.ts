import personalityData from "./aiPersonalities.json";

export const AI_STYLE_IDS = [
  "doctrinal",
  "constrictor",
  "rupture",
  "blizzard",
  "librarian",
  "swarm",
  "fortress"
] as const;

export type AiStyleId = typeof AI_STYLE_IDS[number];

export interface AiPersonalityTraits {
  freezeUrgency?: number;
  selfPreservation?: number;
  libertyBalance?: number;
  containment?: number;
  territory?: number;
  fragmentation?: number;
  development?: number;
  earlyStone?: number;
  structure?: number;
  mobility?: number;
  sacrificeTolerance?: number;
}

export interface AiPersonalityProfile {
  id: AiStyleId;
  label: string;
  goal: string;
  signals: string;
  traits: AiPersonalityTraits;
}

const profiles = personalityData as AiPersonalityProfile[];

if (profiles.map(({ id }) => id).join(",") !== AI_STYLE_IDS.join(",")) {
  throw new Error("AI personality data must contain every style in stable native-code order.");
}

export const AI_STYLE_LIST: readonly AiPersonalityProfile[] = Object.freeze(
  profiles.map((profile) => Object.freeze({
    ...profile,
    traits: Object.freeze({ ...profile.traits })
  }))
);

export const AI_STYLES: Readonly<Record<AiStyleId, AiPersonalityProfile>> = Object.freeze(
  Object.fromEntries(AI_STYLE_LIST.map((profile) => [profile.id, profile]))
) as Readonly<Record<AiStyleId, AiPersonalityProfile>>;

export function aiPersonality(style: string | undefined): AiPersonalityProfile {
  return AI_STYLES[style as AiStyleId] ?? AI_STYLES.doctrinal;
}
