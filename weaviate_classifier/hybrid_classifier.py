"""
Weaviate-based Hybrid Document Classifier
Combines vector similarity search with keyword-based classification
"""

import re
from typing import Dict, List, Any, Tuple
from collections import Counter


class WeaviateHybridClassifier:
    """Hybrid classifier using Weaviate for vector search + keyword matching"""
    
    def __init__(self, weaviate_store):
        self.weaviate_store = weaviate_store
        
        # Document type identifier rules (same as original hybrid classifier)
        self.IDENTIFIER_RULES = {
            "aadhar": [
                r"aadhaar|aadhar|आधार",
                r"unique identification authority",
                r"government of india",
                r"\b\d{4}\s*\d{4}\s*\d{4}\b",  # Aadhaar number pattern
                r"date of birth|dob",
                r"male|female",
                r"address.*pin.*code"
            ],
            "pan_card": [
                r"permanent account number",
                r"income tax department",
                r"pan\s*card",
                r"\b[A-Z]{5}\d{4}[A-Z]\b",  # PAN number pattern
                r"signature|photograph",
                r"father.*name|mother.*name"
            ],
            "passport": [
                r"passport",
                r"republic of india",
                r"type.*p",
                r"country code.*ind",
                r"place of birth",
                r"date of issue|date of expiry",
                r"passport no|document no"
            ],
            "driving_license": [
                r"driving.*licen[cs]e",
                r"transport.*department",
                r"dl.*no|license.*no",
                r"class of vehicle|cov",
                r"date of issue|valid.*till",
                r"blood.*group",
                r"address.*pin"
            ],
            "voter_id": [
                r"election commission",
                r"electoral.*photo.*identity.*card",
                r"epic.*no|voter.*id",
                r"assembly constituency",
                r"part.*no|serial.*no",
                r"age.*as.*on",
                r"father.*name|husband.*name"
            ],
            "bank_statement": [
                r"bank.*statement",
                r"account.*statement",
                r"transaction.*history",
                r"opening.*balance|closing.*balance",
                r"debit|credit",
                r"ifsc.*code",
                r"account.*number|a/c.*no"
            ],
            "salary_slip": [
                r"salary.*slip|pay.*slip",
                r"employee.*id|emp.*id",
                r"basic.*salary|basic.*pay",
                r"gross.*salary|net.*salary",
                r"deductions|allowances",
                r"pf.*number|esi.*number",
                r"pay.*period|salary.*month"
            ],
            "electricity_bill": [
                r"electricity.*bill|electric.*bill",
                r"power.*supply|electricity.*board",
                r"consumer.*number|consumer.*id",
                r"meter.*number|meter.*reading",
                r"units.*consumed|kwh",
                r"due.*date|bill.*date",
                r"tariff|rate"
            ],
            "water_bill": [
                r"water.*bill|water.*tax",
                r"water.*supply|water.*department",
                r"consumer.*number|connection.*id",
                r"meter.*reading|water.*consumption",
                r"due.*date|bill.*period",
                r"cubic.*meter|liters|gallons"
            ],
            "gas_bill": [
                r"gas.*bill|lpg.*bill",
                r"gas.*connection|cylinder",
                r"consumer.*number|customer.*id",
                r"subsidy|refill",
                r"booking.*id|delivery",
                r"kg|cylinder.*capacity"
            ],
            "property_tax": [
                r"property.*tax|house.*tax",
                r"municipal.*corporation|municipality",
                r"assessment.*number|property.*id",
                r"annual.*value|rateable.*value",
                r"tax.*period|financial.*year",
                r"owner.*name|property.*owner"
            ],
            "income_tax_return": [
                r"income.*tax.*return|itr",
                r"assessment.*year|financial.*year",
                r"total.*income|taxable.*income",
                r"tax.*paid|tax.*payable",
                r"acknowledgment.*number|itr.*v",
                r"verification|digital.*signature"
            ],
            "gst_return": [
                r"gst.*return|goods.*services.*tax",
                r"gstin|gst.*identification",
                r"gstr.*[0-9]|gst.*form",
                r"tax.*period|return.*period",
                r"taxable.*value|tax.*amount",
                r"input.*tax.*credit|itc"
            ],
            "GSTR_9": [
                r"gstr.*9|gstr-9",
                r"annual.*return",
                r"consolidated.*return",
                r"financial.*year.*\d{4}.*\d{4}",
                r"gstin.*\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]",
                r"outward.*supplies|inward.*supplies",
                r"tax.*liability|input.*tax.*credit"
            ],
            "GSTR_9C": [
                r"gstr.*9c|gstr-9c",
                r"reconciliation.*statement",
                r"chartered.*accountant|ca.*certificate",
                r"audited.*annual.*financial.*statement",
                r"difference.*amount|reconciliation.*difference",
                r"certified.*that|ca.*certification"
            ],
            "form_16": [
                r"form.*16|form.*no.*16",
                r"certificate.*under.*section.*203",
                r"tax.*deducted.*at.*source|tds",
                r"employer.*details|employee.*details",
                r"pan.*of.*deductor|tan.*number",
                r"salary.*income|income.*from.*salary"
            ],
            "form_26as": [
                r"form.*26as|form.*no.*26as",
                r"annual.*information.*statement|ais",
                r"tax.*credit.*statement",
                r"tds.*credit|advance.*tax",
                r"challan.*identification.*number|cin",
                r"deductor.*tan|deductee.*pan"
            ],
            "lease_agreement": [
                r"lease.*agreement|rent.*agreement",
                r"lessor|lessee",
                r"monthly.*rent|rental.*amount",
                r"security.*deposit|advance",
                r"lease.*period|tenancy.*period",
                r"premises.*address|property.*address"
            ],
            "sale_deed": [
                r"sale.*deed|conveyance.*deed",
                r"vendor|purchaser",
                r"consideration.*amount|sale.*price",
                r"sub.*registrar|registration",
                r"property.*description|survey.*number",
                r"stamp.*duty|registration.*fee"
            ],
            "title_deed": [
                r"title.*deed|property.*title",
                r"ownership.*document|title.*document",
                r"registered.*owner|absolute.*owner",
                r"property.*boundaries|survey.*settlement",
                r"revenue.*records|land.*records",
                r"mutation.*entry|khata.*number"
            ],
            "80g_certificate": [
                r"80g.*certificate|section.*80g",
                r"income.*tax.*act.*1961",
                r"charitable.*institution|charitable.*trust",
                r"donation.*receipt|contribution.*receipt",
                r"eligible.*for.*deduction|tax.*exemption",
                r"registration.*number.*under.*section"
            ],
            "appointment_letter": [
                r"appointment.*letter|offer.*letter",
                r"we.*are.*pleased.*to.*offer|we.*offer.*you",
                r"position|designation|job.*title",
                r"salary|compensation|ctc",
                r"joining.*date|date.*of.*joining",
                r"terms.*and.*conditions|employment.*terms"
            ],
            "experience_letter": [
                r"experience.*letter|experience.*certificate",
                r"to.*whom.*it.*may.*concern",
                r"worked.*with.*us|employed.*with",
                r"period.*of.*employment|duration.*of.*service",
                r"designation|position.*held",
                r"satisfactory.*performance|good.*conduct"
            ],
            "relieving_letter": [
                r"relieving.*letter|relief.*letter",
                r"last.*working.*day|date.*of.*relieving",
                r"cleared.*all.*dues|no.*dues",
                r"wish.*you.*success|best.*wishes",
                r"employment.*terminated|services.*terminated",
                r"handover.*responsibilities|transition"
            ],
            "balance_sheet": [
                r"balance.*sheet",
                r"assets.*and.*liabilities|assets.*&.*liabilities",
                r"current.*assets|fixed.*assets",
                r"current.*liabilities|long.*term.*liabilities",
                r"shareholders.*equity|capital.*and.*reserves",
                r"as.*on.*march.*31|as.*at.*march.*31"
            ],
            "profit_loss": [
                r"profit.*and.*loss|profit.*&.*loss|p.*&.*l",
                r"income.*statement|statement.*of.*income",
                r"revenue|turnover|sales",
                r"cost.*of.*goods.*sold|operating.*expenses",
                r"gross.*profit|net.*profit",
                r"for.*the.*year.*ended"
            ],
            "cash_flow": [
                r"cash.*flow.*statement|statement.*of.*cash.*flows",
                r"operating.*activities|investing.*activities|financing.*activities",
                r"cash.*generated.*from.*operations",
                r"net.*cash.*flow|cash.*and.*cash.*equivalents",
                r"beginning.*of.*year|end.*of.*year"
            ],
            "accounts_payable": [
                r"accounts.*payable|creditors.*list",
                r"trade.*payables|sundry.*creditors",
                r"amount.*payable|outstanding.*amount",
                r"vendor.*name|supplier.*name",
                r"due.*date|payment.*terms",
                r"aging.*analysis|overdue.*amount"
            ],
            "accounts_receivable": [
                r"accounts.*receivable|debtors.*list",
                r"trade.*receivables|sundry.*debtors",
                r"amount.*receivable|outstanding.*receivables",
                r"customer.*name|debtor.*name",
                r"invoice.*date|due.*date",
                r"aging.*analysis|overdue.*receivables"
            ],
            "invoice": [
                r"invoice|bill.*of.*supply",
                r"invoice.*number|invoice.*no|bill.*no",
                r"invoice.*date|bill.*date",
                r"quantity|rate|amount",
                r"tax.*invoice|commercial.*invoice",
                r"total.*amount|grand.*total"
            ],
            "purchase_order": [
                r"purchase.*order|po.*number",
                r"vendor|supplier",
                r"delivery.*date|expected.*delivery",
                r"item.*description|product.*description",
                r"unit.*price|total.*value",
                r"terms.*and.*conditions|payment.*terms"
            ],
            "quotation": [
                r"quotation|quote|price.*quote",
                r"quotation.*number|quote.*no",
                r"valid.*till|validity.*period",
                r"unit.*rate|quoted.*price",
                r"terms.*of.*supply|delivery.*terms",
                r"thank.*you.*for.*your.*inquiry"
            ]
        }
    
    def calculate_keyword_score(self, text: str, doc_type: str) -> float:
        """Calculate keyword matching score for a document type"""
        if doc_type not in self.IDENTIFIER_RULES:
            return 0.0
        
        text_lower = text.lower()
        rules = self.IDENTIFIER_RULES[doc_type]
        matches = 0
        
        for rule in rules:
            if re.search(rule, text_lower):
                matches += 1
        
        return (matches / len(rules)) * 100 if rules else 0.0
    
    def get_all_keyword_scores(self, text: str) -> Dict[str, float]:
        """Get keyword scores for all document types"""
        scores = {}
        for doc_type in self.IDENTIFIER_RULES:
            scores[doc_type] = self.calculate_keyword_score(text, doc_type)
        return scores
    
    def classify(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify document using hybrid approach:
        1. Vector similarity search
        2. Keyword matching (if needed)
        3. Consensus detection
        """
        try:
            # Step 1: Vector similarity search
            vector_results = self.weaviate_store.search_similar(text, limit=top_k)
            
            if not vector_results:
                return {
                    "document_type": "unknown",
                    "confidence": 0.0,
                    "method": "no_results",
                    "details": "No similar documents found in database"
                }
            
            # Extract document types from top results
            top_types = [result["label"] for result in vector_results]
            type_counts = Counter(top_types)
            
            # Step 2: Check for consensus (3+ same type in top 5)
            most_common_type, most_common_count = type_counts.most_common(1)[0]
            
            if most_common_count >= 3:
                # Consensus found - skip keyword matching
                confidence = vector_results[0]["similarity"] * 100
                
                # Prepare top 3 matches for display
                top_3_matches = []
                for i, result in enumerate(vector_results[:3]):
                    top_3_matches.append({
                        "rank": i + 1,
                        "type": result["label"],
                        "vector_score": round(result["similarity"] * 100, 2),
                        "keyword_score": 0,  # Not calculated due to consensus
                        "combined_score": round(result["similarity"] * 100, 2)
                    })
                
                return {
                    "document_type": most_common_type,
                    "confidence": round(confidence, 1),
                    "method": "consensus",
                    "vector_score": round(confidence, 1),
                    "keyword_score": 0,
                    "details": f"Consensus detected: {most_common_count}/{top_k} matches are '{most_common_type}'",
                    "top_3_matches": top_3_matches
                }
            
            # Step 3: No consensus - apply keyword matching to top candidates
            unique_types = list(set(top_types))
            keyword_scores = {}
            
            for doc_type in unique_types:
                keyword_scores[doc_type] = self.calculate_keyword_score(text, doc_type)
            
            # Find best keyword match
            best_keyword_type = max(keyword_scores.keys(), key=lambda x: keyword_scores[x])
            best_keyword_score = keyword_scores[best_keyword_type]
            
            # Get vector score for the best keyword match
            best_vector_score = 0
            for result in vector_results:
                if result["label"] == best_keyword_type:
                    best_vector_score = result["similarity"] * 100
                    break
            
            # Determine final result
            if best_keyword_score > 30:  # Strong keyword match threshold
                final_type = best_keyword_type
                final_confidence = best_keyword_score
                method = "keyword"
                details = f"Strong keyword match ({best_keyword_score:.1f}%)"
            else:
                # Fall back to vector similarity
                final_type = vector_results[0]["label"]
                final_confidence = vector_results[0]["similarity"] * 100
                method = "vector"
                details = f"Vector similarity (keyword score too low: {best_keyword_score:.1f}%)"
            
            # Prepare top 3 matches with both scores
            top_3_matches = []
            for i, result in enumerate(vector_results[:3]):
                doc_type = result["label"]
                vector_score = result["similarity"] * 100
                keyword_score = keyword_scores.get(doc_type, 0)
                combined_score = max(vector_score, keyword_score)  # Take the higher score
                
                top_3_matches.append({
                    "rank": i + 1,
                    "type": doc_type,
                    "vector_score": round(vector_score, 2),
                    "keyword_score": round(keyword_score, 2),
                    "combined_score": round(combined_score, 2)
                })
            
            return {
                "document_type": final_type,
                "confidence": round(final_confidence, 1),
                "method": method,
                "vector_score": round(best_vector_score, 1),
                "keyword_score": round(best_keyword_score, 1),
                "details": details,
                "top_3_matches": top_3_matches
            }
            
        except Exception as e:
            return {
                "document_type": "unknown",
                "confidence": 0.0,
                "method": "error",
                "details": f"Classification error: {str(e)}"
            }
    
    def get_keyword_analysis(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get keyword analysis for all document types"""
        scores = self.get_all_keyword_scores(text)
        # Sort by score descending and return top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]