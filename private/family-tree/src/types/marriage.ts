export type RelationType = "marriage" | "parenthood";

export type RelationPlacement = "same" | "parent" | "ancestor" | "child";

export type RelationDetails = {
  role?: "father" | "mother";
  parentIsSource?: boolean;
  spacing?: number;
  registrationDate?: string;
  registrationPlace?: string;
  registrationAuthority?: string;
  recordNumber?: string;
  status?: "active" | "dissolved";
  dissolutionDate?: string;
  dissolutionPlace?: string;
  dissolutionAuthority?: string;
  notes?: string;
  couplePhotoUrl?: string;
  documents: string[];
};

export type RelationData = {
  id: string;
  type: RelationType;
  personAId: string;
  personBId: string;
  details?: Record<string, unknown>;
};

export type MarriageData = RelationData;
