# GraphGuard - AI-Powered Dynamic Authentication System
## Week 4 Capstone Integration Submission



- **Deployed App:** https://graphguard1.streamlit.app/


### Project Overview
GraphGuard is an intelligent authentication and fraud detection system that uses AI to generate personalized security challenges based on real transaction history. The system provides:

- **Dynamic Authentication**: AI-generated questions specific to each user's transaction history
- **Real-time Fraud Detection**: Pattern analysis and risk scoring for suspicious transactions
- **Comprehensive Monitoring**: Automatic logging of all interactions with performance metrics
- **Production-Ready Architecture**: Scalable design with fallback mechanisms and monitoring



### Files Included

#### 1. `streamlit_app.py` - Main Application
Updated Streamlit application with:
- CSV transaction data loading (uses your actual transaction file)
- Comprehensive logging system (`logs/product_metrics.csv`)
- Real-time monitoring dashboard
- Authentication and fraud detection modules
- Fallback to demo data if CSV not found

#### 2. `Week4_Deliverables.docx` - Complete Documentation
Comprehensive capstone documentation including:
- **Section A**: Capstone Product Integration Brief
- **Section B**: Application Integration details
- **Section C**: Monitoring & Logging implementation
- **Section D**: Capstone Evaluation with impact metrics
- **Section E**: Deployment Readiness Plan with architecture
- **Section F**: Failure scenarios and mitigation strategies

#### 3. `logs/product_metrics.csv` - Sample Logs
Contains 10 example logged interactions demonstrating:
- Authentication generation events
- Authentication verification events
- Fraud detection analysis events
- Latency measurements
- Confidence scores
- Success/failure status

### Quick Start Guide

#### Prerequisites
```bash
pip install streamlit pandas plotly anthropic
```

#### Running the Application

**Option 1: Use Your Transaction CSV**
```bash
# Place your CSV file in the same directory
streamlit run streamlit_app.py
```
Then in the sidebar:
1. Select "Load CSV File"
2. Enter filename: `transactions_export_20260213_000349.csv`
3. Click "Load Data"

**Option 2: Use Demo Data**
```bash
streamlit run streamlit_app.py
```
Then in the sidebar:
1. Select "Demo Data"
2. System automatically generates synthetic transactions

#### Optional: Enable Full AI Features
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```
*Note: System works in demo mode without API key*

### Application Features

#### Tab 1: Authentication 🔑
1. Select a user from dropdown
2. Choose security level (LOW/MEDIUM/HIGH/CRITICAL)
3. Click "Generate Questions" to create AI-powered challenges
4. Answer the questions and submit
5. View authentication result and score

#### Tab 2: Fraud Detection 🚨
1. Select a user
2. Choose a transaction from their history
3. Click "Analyze with AI"
4. View fraud score, risk level, insights, and recommendations

#### Tab 3: Dashboard 📊
- System-wide statistics
- Transaction distribution by category
- Daily volume trends
- Top merchant analysis

#### Tab 4: Explorer 🔍
- Filter transactions by user, category, location, amount
- Export filtered data to CSV
- Explore complete transaction dataset

### Monitoring & Logs

#### Log File Location
`logs/product_metrics.csv`

#### Logged Fields
- **timestamp**: ISO 8601 timestamp
- **user_task_type**: Operation type (authentication_generation, authentication_verification, fraud_detection)
- **user_id**: User identifier
- **retrieval_configuration**: Security level or analysis type
- **latency_ms**: Response time in milliseconds
- **evidence_ids**: Transaction IDs used as evidence
- **confidence_score**: Model confidence (0.0-1.0)
- **faithfulness_indicator**: Quality indicator (high/medium/low/error)
- **status**: Operation result (success/failed)

#### Viewing Logs
The sidebar shows:
- Total interactions logged
- System success rate
- Log file path

### Architecture

```
User Interface (Streamlit)
        ↓
API Gateway / Load Balancer
        ↓
    ┌───────────────┬───────────────┐
    ↓               ↓               ↓
Authentication  Fraud Detection  Logging
Module          Engine           System
    ↓               ↓               ↓
    └───────────────┴───────────────┘
                    ↓
        Transaction Database
                    ↓
        Monitoring & Alerts
```

### Data Schema

#### Transaction CSV Format
```
TRANSACTION_ID,USER_ID,MERCHANT_NAME,CATEGORY,AMOUNT,LOCATION,TRANSACTION_DATE,STATUS
862E6A0C5BC9,USER_008,CVS,Retail,294.94,"Chicago, IL",2026-02-12 17:46:36,COMPLETED
```

#### Supported Categories
- Coffee Shops
- Gas Stations
- Restaurants
- Grocery
- Electronics
- Jewelry
- Retail
- Clothing

### Deployment Notes

#### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

#### Production Deployment

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets (ANTHROPIC_API_KEY)
4. Deploy

**AWS/Azure:**
1. Containerize with Docker
2. Deploy to ECS/Fargate or App Service
3. Configure environment variables
4. Set up monitoring (CloudWatch/Application Insights)

**Scaling Considerations:**
- Use PostgreSQL for transaction database (not CSV)
- Implement Redis caching for frequent queries
- Load balancer for multiple app instances
- Auto-scaling based on CPU/memory metrics

### Security & Compliance

#### Implemented Security Features
- API key authentication for Claude API
- Rate limiting (configurable)
- PII data handling (transaction IDs only)
- Secure logging (no sensitive data exposure)

#### Required for Production
- TLS/SSL encryption for all traffic
- Database encryption at rest
- RBAC for admin functions
- Audit logging for compliance
- GDPR/PCI-DSS compliance measures

### Troubleshooting

#### Issue: CSV File Not Found
**Solution:** 
- Ensure CSV file is in same directory as app
- Check filename matches exactly (case-sensitive)
- Or use "Demo Data" mode for testing

#### Issue: AI Questions Not Generating
**Solution:**
- System automatically falls back to demo mode
- Add ANTHROPIC_API_KEY for full AI features
- Check API key is valid and has credits

#### Issue: Slow Performance
**Solution:**
- Use data caching (already implemented)
- Reduce transaction history size (set to last 10-20)
- Deploy on more powerful instance

#### Issue: Logs Not Creating
**Solution:**
- Check write permissions on logs directory
- Verify logs folder exists
- System creates automatically on first run

### Performance Metrics

#### Target Performance
- Authentication latency: <500ms (P95)
- Fraud detection latency: <1000ms (P95)
- Success rate: >95%
- System uptime: 99.9%

#### Actual Performance (from logs)
- Average authentication latency: ~380ms
- Average fraud detection latency: ~880ms
- Success rate: 100% (10/10 logged interactions)
- All operations completed successfully

### Week 4 Deliverables Checklist

✅ **A. Integration Brief** - See Week4_Deliverables.docx Section A  
✅ **B. Application Integration** - streamlit_app.py with all components  
✅ **C. Monitoring & Logging** - logs/product_metrics.csv with 10+ entries  
✅ **D. Evaluation** - Impact metrics and technical analysis in document  
✅ **E. Deployment Plan** - Architecture diagram and scaling strategy  
✅ **F. Failure Analysis** - "Silent Drift" scenario with mitigation  
✅ **Individual Reflection** - Template provided in document  

### Next Steps

#### For Assignment Submission
1. **GitHub Repository:**
   - Upload all files to your repository
   - Include README.md (this file)
   - Push transaction CSV and logs folder
   - Tag release as `week-4-submission`

2. **Deploy Application:**
   - Deploy to Streamlit Cloud or AWS
   - Get public URL
   - Update document with deployment link
   - Test all features on deployed version

3. **Complete Individual Reflection:**
   - Open Week4_Deliverables.docx
   - Navigate to "Individual Reflection" section
   - Write 1 paragraph about production readiness improvements
   - Save and include in submission

4. **Canvas Submission:**
   - GitHub repository link
   - Deployed application link
   - Week4_Deliverables.docx (if separate from repo)




---

### Related Files
- Original assignment: CS5588_Week4_HandsOn.docx
- Transaction data: transactions_export_20260213_000349.csv
- Sample logs: logs/product_metrics.csv


