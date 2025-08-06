import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from analyzer import BankruptcyAwareFinBERTAnalyzer  
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Bankruptcy Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: black;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
   
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4ECDC4;
    }
   
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
   
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
   
    .neutral-sentiment {
        color: #6c757d;
        font-weight: bold;
        font-size: 1.5rem;
    }
   
    .complexity-high {
        color: #ff6b6b;
        font-weight: bold;
    }
   
    .complexity-medium {
        color: #ffa726;
        font-weight: bold;
    }
   
    .complexity-low {
        color: #66bb6a;
        font-weight: bold;
    }
   
    .stMetric > div > div > div > div {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzer():
    """Load the bankruptcy analyzer model"""
    try:
        analyzer = BankruptcyAwareFinBERTAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"Error loading analyzer: {e}")
        return None

def get_company_news(company_name, num_articles=5):
    """Fetch recent news about the company (hardcoded mock data for each)"""

    all_news = {
        'Ascena Retail Group': [
            {
                'title': 'Ascena Retail Group Reports Q3 Financial Results',
                'description': 'Company announces quarterly earnings with mixed results...',
                'url': 'https://www.businessoffashion.com/organisations/ascena-retail-group/',
                'publishedAt': '2024-01-15T10:30:00Z',
                'source': 'Business of Fashion'
            },
            {
                'title': 'Ascena Retail Group Announces Strategic Restructuring Plan',
                'description': 'Store closures and workforce reduction ahead...',
                'url': 'https://www.wealthmanagement.com/real-estate/ann-taylor-parent-goes-bankrupt-plans-to-shut-over-1-000-stores',
                'publishedAt': '2024-01-10T14:20:00Z',
                'source': 'Wealth Management'
            },
        ],
        'American Apparel': [
            {
                'title': 'JCPenney Looks to Reinvent Itself in 2024',
                'description': 'New CEO sets bold vision to reposition JCPenney in the retail market...',
                'url': 'https://example.com/jcpenney-reinvention',
                'publishedAt': '2024-02-01T09:30:00Z',
                'source': 'Retail Dive'
            },
            {
                'title': 'JCPenney Expands Partnership with Influencers',
                'description': 'Brand taps into Gen Z and millennial demographics through social media campaigns...',
                'url': 'https://example.com/jcpenney-influencers',
                'publishedAt': '2024-01-25T12:00:00Z',
                'source': 'AdWeek'
            },
        ],
        'Bed, Bath and Beyond': [
            {
                'title': 'Macy\'s Announces Store Modernization Plan',
                'description': 'Investing in in-store tech to enhance customer experience...',
                'url': 'https://example.com/macys-modernization',
                'publishedAt': '2024-01-28T08:45:00Z',
                'source': 'CNBC'
            },
            {
                'title': 'Macy\'s Holiday Season Performance Beats Expectations',
                'description': 'Strong online sales drive growth despite economic pressures...',
                'url': 'https://example.com/macys-holiday-sales',
                'publishedAt': '2024-01-18T15:30:00Z',
                'source': 'Forbes'
            },
        ]
    }

    return all_news.get(company_name, [])[:num_articles]

# Mock company data for dropdown
company_data = {
    "Ascena Retail Group": """
    The Debtors have filed the Chapter 11 Cases to implement the terms of a Restructuring Support Agreement, dated July 23, 2020 (together with all exhibits and schedules thereto, the “RSA”), by and among the Company and certain of its subsidiaries (each, a “Company Party” and collectively, the “Company Parties”) and members of an ad hoc group of lenders (the “Consenting Stakeholders”) under the Term Credit Agreement, dated as of August 21, 2015 (as amended, restated, supplemented or otherwise modified from time to time, the “Prepetition Term Credit Agreement”), among the Company, AnnTaylor Retail, Inc., the lenders party thereto and Goldman Sachs Bank USA, as administrative agent. The RSA is supported by Consenting Stakeholders holding approximately 68% of the borrowings under the Prepetition Term Credit Agreement as of the Petition Date.

    As part of the Chapter 11 Cases, the Company has divested the Catherines’ E-Commerce business and Justice’s intellectual property and other assets. The Company has also entered into an agreement to sell assets relating to the Company’s Ann Taylor, LOFT and Lane Bryant brands. See Note 24 to the accompanying consolidated financial statements.

    For the duration of the Chapter 11 Cases, the Company’s operations and ability to develop and execute its business plan are subject to the risks and uncertainties associated with the Chapter 11 process as described in Part I, Item 1A — “Risk Factors” of this Annual Report on Form 10-K. As a result of these risks and uncertainties, the amount and composition of the Company’s assets and liabilities could be significantly different following the outcome of the Chapter 11 Cases, and the description of the Company’s operations, properties and liquidity and capital resources included in this Annual Report on Form 10-K may not accurately reflect its operations, properties and liquidity and capital resources following emergence from the Chapter 11 Cases.

    For more information regarding the Chapter 11 Cases, see Note 2 to the accompanying consolidated financial statements, and for information regarding our ability to continue as a going concern, see Note 1 to the accompanying consolidated financial statements.

    COVID-19 Pandemic

    As described elsewhere herein, the coronavirus disease ("COVID-19") has had far-reaching adverse impacts on many aspects of our operation, directly and indirectly, including our employees, consumer behavior, distribution and logistics, our suppliers, and the market overall. The scope and nature of these impacts have been rapidly changing and is expected to continue to be so in the near term. In light of the continued uncertain situation relating to COVID-19, we took a number of precautionary measures in the second half of Fiscal 2020 to manage our resources conservatively by reducing and/or deferring capital expenditures, inventory purchases and operating expenses to mitigate the adverse impact of COVID-19, which is intended to help minimize the risk to our Company, employees, customers, and the communities in which we operate. Such measures include the following:
    •The temporary closure of our retail stores;
    •The temporary furlough of a substantial portion of our workforce;
    •Reductions in pay of ranging amounts for a substantial majority of those employees not placed on temporary furlough;
    •Working with our landlords to minimize costs associated with closed retail stores;
    •Drawing down $230 million under the Company’s revolving credit facility as a precautionary measure in order to increase its cash position and preserve financial flexibility; and
    •Extended vendor payment terms on merchandise and non-merchandise purchases.

    In addition to the effects described above, our supply chain has been affected by COVID-19. Certain of the Company’s vendors in Asia were temporarily closed for a portion of the third quarter of Fiscal 2020 as a result of COVID-19, however the vendors had resumed production by the end of the quarter. It is possible that if COVID-19 re-emerges in the countries where we obtain our goods, it could cause our vendors to cease production again. At the current time, we believe that we have sufficient inventory and supplies to support our demand in the near future.

    Besides the lower store sales impact, COVID-19 also significantly impacted our margin rates and our long-term growth assumptions, which resulted in long-term asset impairments in the second half of Fiscal 2020, as described more fully in Notes 7 and 10 to the accompanying consolidated financial statements.
    In May 2020, the Company started to reopen its retail stores. As stores began to reopen, the Company also began to bring certain employees back from temporary furlough. Employees necessary to support the phased re-opening of the business were brought back first. In addition, in June 2020, the Company restored all employees who were not on temporary furlough back to their original pay rates.

    Although each of the remedial measures discussed above were taken by the Company to protect the business and preserve liquidity, each may also have the potential to have a material adverse impact on our current business, financial condition and results of operations, and may create additional risks for our Company. While we started to reverse at least some of the temporary measures, we cannot predict the specific duration for which other precautionary measures will stay in effect, and we may elect or need to take additional measures, or reinstate previous measures, as the information available to us continues to develop, including with respect to our employees, distribution centers, relationships with our third-party vendors, and our customers.

    As the Company reopened its retail stores, it did so in accordance with local government guidelines. As of the time of this filing, substantially all of our retail stores have re-opened to the public with restricted operations. However, the Company continues to closely monitor changes in government guidelines and of the outbreak itself. In certain cases, we have had to close stores that had re-opened. As a result, we continue to believe that COVID-19 will have a significant negative impact on our results of operations, financial position and cash flows through the first half of Fiscal 2021.

    Seasonality of Business

    Our individual segments are typically affected by seasonal sales trends primarily resulting from the timing of holiday and back-to-school shopping periods. In particular, sales at our Kids Fashion segment tend to be significantly higher during the Fall season, which occurs during the first and second quarters of our fiscal year, as this includes the back-to-school period and the December holiday season. Our Plus Fashion segment tends to experience higher sales during the Spring season, which include the Easter and Mother's Day holidays. Our Premium Fashion segment has relatively balanced sales across the Fall and Spring seasons. As a result, our operational results and cash flows may fluctuate materially in any quarterly period depending on, among other things, increases or decreases in comparable store sales, adverse weather conditions, shifts in the timing of certain holidays and changes in merchandise mix.
   
    Summary of Financial Performance

    Discontinued Operations
   
    Dressbarn Wind Down

    The Company completed the wind down of its Dressbarn brand during the second quarter of Fiscal 2020. All Dressbarn store locations were closed as of December 31, 2019. As a result, the Company's Dressbarn business has been classified as a component of discontinued operations within the audited consolidated financial statements for all periods presented, and when coupled with the sale of maurices discussed below, we no longer present results of the Value Fashion segment. The operating results of Dressbarn are excluded from the discussion below. In connection with the Dressbarn wind down, we have incurred cumulative costs of approximately $58 million, of which approximately $5 million was incurred during Fiscal 2020 and included in discontinued operations.

    Sale of maurices

    On May 6, 2019, the Company and Maurices Incorporated, a Delaware corporation (“maurices”) and wholly owned subsidiary of ascena, completed the transaction contemplated by the previously-announced Stock Purchase Agreement with Viking Brand Upper Holdings, L.P., a Cayman Islands exempted limited partnership (“Viking”) and an affiliate of OpCapita LLP, providing for, among other things, the sale by ascena of maurices to Viking (the “Transaction”). Effective upon the closing of the Transaction in May 2019, ascena received cash proceeds of approximately $210 million and a 49.6% ownership interest in the operations of maurices, consisting of interests in Viking preferred and common stock. As discussed in Note 3 to the accompanying consolidated financial statements, upon completion of the sale of maurices, the Company has classified

    maurices as a component of discontinued operations within the audited consolidated financial statements for Fiscal 2019 and is also excluded from the discussion below.

    Goodwill and Other Indefinite-lived Intangible Asset Impairment Charges

    While overall performance during the second quarter of Fiscal 2020 was in line with management's expectations, lower than expected sales at our Justice brand, and lower than expected margins at our Ann Taylor brand, resulted in a conclusion that these factors represented impairment indicators which required the Company to test our goodwill and indefinite-lived intangible assets for impairment during the second quarter of Fiscal 2020. As a result of the assessment, we recognized goodwill impairment charges of $54.9 million and $8.5 million at the Ann Taylor and Justice reporting units, respectively, to write-down the carrying values of the reporting units to their fair values. In addition, we recorded non-cash impairment charges to write-down the carrying values of our other intangible assets of $46.9 million which substantially consisted of write-downs of our trade name intangible assets to their fair values at Ann Taylor and Justice by $10.0 million and $35.0 million, respectively.

    Further, the impact of the retail store closures in response to COVID-19, along with the continued declines in the stock price and the fair value of our Term Loan debt, resulted in a conclusion that another triggering event occurred in the third quarter of Fiscal 2020, thereby requiring us to test our goodwill and intangible assets for impairment (the “April Interim Test”). The April Interim Test reflected revised long-range assumptions that were reflected in contemplation of the Chapter 11 Cases discussed above. Those assumptions included the wind down of the Catherines brand, and a significant reduction in the number of Justice retail stores, an overall reductions in the number of retail stores at the Company's other brands, and a significant reduction in the Company's workforce commensurate with the store reductions. As a result of the revised assumptions, we recognized goodwill impairment charges of $15.0 million and $70.5 million at the Ann Taylor and LOFT reporting units to write-down the carrying value of the reporting units to their fair value. In addition, we recorded non-cash impairment charges totaling $41.3 million to write-down the carrying values of our trade name intangible assets to their fair values as follows: $17.7 million at Ann Taylor, $7.8 million at LOFT, $7.8 million at Justice, $3.0 million at Catherines and $5.0 million of our Justice international franchise rights. These impairment charges are more fully described in Note 7 to the accompanying consolidated financial statements.

    Finally, the commencement of the Chapter 11 Cases, discussed above, led the Company to conclude this represented further impairment indicators. As a result, the Company was required to test its goodwill and indefinite-lived intangible assets for impairment during the fourth quarter of Fiscal 2020 (the “Year-End Test"). The cash flow projections underlying the Year-End Test reflected revised assumptions for Fiscal 2021 and Fiscal 2022 based on the current consumer demand, the continuation of the negative impacts of COVID-19 and the potential release of a vaccine during Fiscal 2021 with the expectation that the Company will be back in line with the original long-range projections for Fiscal 2023 and beyond. As of the third quarter of Fiscal 2020, only our LOFT brand had goodwill remaining and was tested for goodwill impairment in the fourth quarter. The Year-End Test indicated the fair value of the LOFT brand exceeded its carrying value and as a result no goodwill impairment charge was required in the fourth quarter. However, the Year-End Test resulted in the Company recognizing impairment charges to write-down the carrying values of its other intangible assets to their fair values as follows: $34.2 million on the LOFT trade name and $1.3 million on the Ann Taylor trade name. In addition, the Company impaired the remaining Justice international franchise rights of $5 million as the Company will no longer be supporting this business strategy.

    Tangible Asset Impairment Charges

    As a result of the revised projections utilized in the Company’s goodwill and intangible asset impairment testing described above, which reflect significant reductions in near-term cash flows of certain of our retail stores, as well as the planned store reductions discussed above, we recognized impairment charges of $196.8 million to write-down store-related fixed assets and right-of-use assets to their fair values. Impairment charges by segment reflected $115.3 million at the Premium Fashion segment, $25.2 million at the Plus Fashion segment, and $56.3 million at the Kids Fashion segment. In addition, a long-lived Corporate asset impairment charge of $12.9 million was recorded during the second half of Fiscal 2020 which reflects an $8.4 million write-down of the book value of the Company’s campus in Mahwah, NJ to fair market value recorded in the third quarter of Fiscal 2020 in connection with its planned sale and a long-lived asset impairment charge of $4.5 million was recorded in the fourth quarter of Fiscal 2020 relating to the Company's Duluth, MN building to reflect a write-down of the book value to fair market value. These impairment charges are more fully described in Note 10 to the accompanying consolidated financial statements.

    Operating highlights for Fiscal 2020 are as follows:

    •Sales decreased by 21.5%, reflecting decreases at all of our operating segments due primarily to the temporary retail store closures;
    •Gross margin rate decreased by 530 basis points to 50.6% primarily due to markdown and promotional selling necessary to clear excess inventory that was unable to be sold after the temporary retail store closures;
    •Operating loss of $1,113.6 million compared to $638.3 million for the year-ago period, resulting primarily from the lower Net sales and Gross margin decline, higher store-related asset-impairment charges, and higher restructuring and other related charges, which were offset in part by expense reductions, primarily reflecting the furlough of a significant portion of our workforce in response to COVID-19, the impact of our previously-announced cost reduction initiatives, and lower impairment of goodwill and other intangible assets; and
    •Net loss from continuing operations per diluted share of $120.68 in Fiscal 2020, compared to $73.87 in Fiscal 2019.

    Liquidity for Fiscal 2020 primarily reflected:
   
    •Cash flows provided by operations was $131.6 million, compared to $21.1 million in the year-ago period;
    •Cash flows used in investing activities for Fiscal 2020 was $39.5 million, consisting primarily of capital expenditures of $65.1 million, offset in part by $20.6 million received from the sale of our corporate building in Mahwah, New Jersey, compared to net cash provided by investing activities of $67.7 million in the year-ago period; and
    •Cash flows provided by financing activities for Fiscal 2020 was $160.3 million, consisting primarily of the $230.0 million borrowed under the revolving credit facility during the third quarter, offset in part by the repurchased $79.5 million of outstanding term loan debt for $49.4 million and the term loan prepayment of $20.4 million compared to net cash provided by financing activities of $0.3 million in the year-ago period.

    Net sales. Total Net sales decreased by $1,016.6 million, or 21.5%, to $3,718.1 million. Specifically, the total store and e-commerce revenue decreased by $974.1 million and other revenue decreased by $42.5 million compared to the prior year period. The decrease in Net sales was primarily driven by the temporary closure of our retail stores as a result of COVID-19 during the second half of Fiscal 2020.
    Gross margin. Gross margin, in terms of dollars, was primarily lower as a result of a decline in sales and a decline in rate, which is discussed on a segment basis below. The gross margin rate represents the difference between net sales and cost of goods sold, expressed as a percentage of net sales. Gross margin rate is dependent upon a variety of factors, including brand sales mix, product mix, channel mix, the timing and level of promotional activities and fluctuations in material costs. These factors, among others, may cause cost of goods sold as a percentage of net revenues to fluctuate from period to period.

    Gross margin rate decreased by 530 basis points from 55.9% for Fiscal 2019 to 50.6% for Fiscal 2020 resulting from lower margin at all three operating segments. Gross margin rate highlights on a segment basis are as follows:

    •Premium Fashion gross margin rate performance declined to 51.5% for Fiscal 2020 from 56.5% for Fiscal 2019 primarily reflecting increased inventory reserves and promotional selling to clear excess inventory that was unable to be sold in the normal course due to the temporary closure of our retail stores as a result of COVID-19 during the second half of Fiscal 2020 and higher shipping costs related to increased direct channel penetration.
    •Plus Fashion gross margin rate performance declined to 53.9% for Fiscal 2020 from 57.3% for Fiscal 2019 primarily reflecting inventory markdown reserves and increased promotional selling to clear excess inventory that was unable to be sold in the normal course due to the temporary closure of our retail stores as a result of COVID-19 during the second half of Fiscal 2020 and higher shipping costs related to increased direct channel penetration, offset in part by the improved product acceptance experienced during the first half of Fiscal 2020.
    Kids Fashion gross margin rate performance declined to 44.3% for Fiscal 2020 from 53.0% for Fiscal 2019 primarily due to significant inventory markdown reserves and increased promotional selling to clear excess inventory that was unable to be sold in the normal course due to the temporary closure of our retail stores as a result of COVID-19 during the second half of Fiscal 2020. The gross margin decline also reflects increased markdowns in the first half of Fiscal 2020 resulting from lower store traffic.

    Buying, distribution and occupancy ("BD&O") expenses consist of store occupancy and utility costs (excluding depreciation) and all costs associated with the buying and distribution functions.
   
    BD&O expenses decreased by $128.0 million, or 13.4%, to $825.8 million in Fiscal 2020. The reduction in expenses was driven by the furlough in the second half of Fiscal 2020 of a significant portion of our workforce in response to COVID-19, lower occupancy expense and lower-employee related costs, both resulting from the continued impact of our previously announced cost reduction efforts, as well as amounts received under the transition services agreement with maurices as further discussed in Note 11 to the accompanying consolidated financial statements. BD&O expenses as a percentage of net sales increased to 22.2% in Fiscal 2020 from 20.1% in Fiscal 2019.

    Selling, general and administrative (“SG&A”) expenses consist of compensation and benefit-related costs for sales and store operations personnel, administrative personnel and other employees not associated with the functions described above under BD&O expenses. SG&A expenses also include advertising and marketing costs, information technology and communication costs, supplies for our stores and administrative facilities, insurance costs, legal costs and costs related to other administrative services.
   
    SG&A expenses decreased by $124.7 million, or 8.1%, to $1,421.1 million in Fiscal 2020. The decrease in SG&A expenses was primarily due to the furlough in the second half of Fiscal 2020 of a significant portion of our workforce in response to COVID-19, the continuation of our previously announced cost reduction initiatives, mainly reflecting lower store related expenses and non-merchandise procurement savings, lower marketing expenses, and amounts received under the transition services agreement with maurices, as further discussed in Note 11 to the accompanying consolidated financial statements. These savings were offset in part by higher store-related impairment charges. SG&A expenses as a percentage of net sales increased to 38.2% in Fiscal 2020 from 32.6% in Fiscal 2019.

    Depreciation and amortization expense decreased by $47.7 million, or 17.0%, to $232.7 million in Fiscal 2020. The decrease was across all of our operating segments and was driven by a lower level of store-related fixed-assets, offset in part by incremental depreciation from capital investments placed into service during Fiscal 2019.

    Operating loss. Operating loss was $1,113.6 million for Fiscal 2020 compared to $638.3 million in Fiscal 2019
    Premium Fashion operating results decreased by $316.0 million primarily driven by a decline in Net sales and a lower gross margin rate, as discussed above, offset in part by lower operating expenses. Operating expense reductions were primarily driven by the furlough in the second half of Fiscal 2020 of a significant portion of our workforce in response to COVID-19, lower expenses as a result of our continued cost reduction efforts, and lower depreciation expense. These expense reductions were offset in part by significantly higher store-related impairment charges.
    Plus Fashion operating results improved by $11.5 million primarily due to lower operating expenses, which were mostly offset by declines in Net sales and a lower gross margin rate, as discussed above. Operating expense reductions were primarily driven by the furlough in the second half of Fiscal 2020 of a significant portion of our workforce in response to COVID-19 and lower expenses as a result of our continued cost reduction efforts.
    Kids Fashion operating results decreased by $159.9 million primarily due to a decline in Net sales and a lower gross margin rate, as discussed above, offset in part by lower operating expenses. Operating expense reductions were primarily driven by the furlough in the second half of Fiscal 2020 of a significant portion of our workforce in response to COVID-19 and lower expenses as a result of our continued cost reduction efforts. These expense reductions were offset in part by significantly higher store-related impairment charges.

    Unallocated restructuring and other related charges of $238.3 million primarily includes charges resulting from announcements and actions taken in connection with the Chapter 11 Cases and included $29.3 million of severance and other related charges, $19.3 million of professional fees, and $189.3 million of non-cash asset impairments primarily related to right-of-use lease assets associated with the stores identified to close as a result of the Chapter 11 Cases. The $94.1 million of unallocated restructuring and other related charges in Fiscal 2019 primarily included $33.1 million for professional fees incurred in connection with the identification and implementation of transformation initiatives, $16.1 million of severance and other related charges, reflecting severance associated with the cost reduction actions taken in the fourth quarter of Fiscal 2019, and $44.9 million of non-cash asset impairments, reflecting the write-down of the Mahwah, NJ corporate headquarters as a result of the Dressbarn wind down and the write-down of a corporate-owned office building in Duluth, MN to fair market value as a result of the sale of maurices.
    Unallocated impairment of goodwill reflects the Fiscal 2020 write-down of the carrying values of the Ann Taylor, LOFT and Justice reporting units to their fair values as follows: $69.9 million at Ann Taylor, $70.5 million at LOFT, and $8.5 million at Justice. The $276.0 million in Fiscal 2019 reflects the write-down of the carrying values of the Lane Bryant, Catherines and Justice reporting units.

    Unallocated impairment of other intangible assets reflects the Fiscal 2020 write-down of the Company's trade name intangible assets to their fair values as follows: $29.0 million of our Ann Taylor trade name, $42.0 million of our LOFT trade name, $42.8 million of our Justice trade name, $4.0 million of our Catherines trade name and $10.9 million of our Justice franchise rights. The $134.9 million in Fiscal 2019 reflects the write-down of the Company's trade name intangible assets to their fair values as follows; $15.0 million of our Ann Taylor trade name, $60.3 million of our LOFT trade name, $37.0 million of our Lane Bryant trade name, $6.0 million of our Catherines trade name and $16.6 million of our Justice trade name.

    Interest expense decreased by $7.6 million to $99.4 million for Fiscal 2020, primarily due to a lower average outstanding term loan balance as a result of the second quarter debt repurchases and a lower interest rate on our variable-rate term loan during Fiscal 2020, partially offset by interest on our revolving borrowings drawn down during the third quarter of Fiscal 2020. There were no revolver borrowings outstanding during Fiscal 2019.

    Gain on extinguishment of debt. In Fiscal 2020, the Company repurchased $79.5 million of outstanding principal balance of the term loan at an aggregate cost of $49.4 million through the open market transactions, resulting in a $28.5 million pre-tax gain, net of the proportional write-off of unamortized original discount and debt issuance costs of $1.6 million. There was no gain on extinguishment of debt in the year-ago period.

    Reorganization items, net of $3.4 million during Fiscal 2020 represent the post-petition costs directly associated with the Chapter 11 Cases. These costs reflect professional fees incurred after the commencement of the Chapter 11 Cases. There were no Reorganization items, net in the year-ago period.
    Cash flows provided by operating activities. Net cash provided by operating activities was $131.6 million for Fiscal 2020, compared with cash provided by of $21.1 million during the year-ago period. The increase in cash flows provided by operating activities in Fiscal 2020 primarily reflected reduced inventory purchases and the extension of payment terms to vendors and landlords, offset in part by lower earnings before non-cash expenses as a result of lower sales and margin rates due to COVID-19.

    Cash flows (used in) provided by investing activities. Net cash used in investing activities for Fiscal 2020 was $39.5 million, compared with cash provided by investing activities of $67.7 million for the year-ago period. Net cash used in investing activities in Fiscal 2020 consisted primarily of capital expenditures of $65.1 million, offset in part by $20.6 million received from the sale of our corporate building in Mahwah, New Jersey and $5.0 million received from the sale of intellectual property rights associated with the Dressbarn direct channel operations. Net cash provided by investing activities in the year-ago period was $67.7 million, consisted primarily of net proceeds from the sale of maurices of $203.2 million, offset in part by capital expenditures of $136.5 million.

    Net cash provided by financing activities. Net cash provided by financing activities was $160.3 million during Fiscal 2020 and reflects proceeds from our revolving credit facility of $230.0 million offset by repurchases of our Term Loan debt of $69.8 million. Net cash provided by financing activities was $0.3 million during the year-ago period.
    """,
    "American Apparel": """
    On December 16, 2014, the Board appointed Paula Schneider as CEO, effective January 5, 2015. This appointment followed the termination of Dov Charney, former President and CEO, for cause in accordance with the terms of his employment agreement. Scott Brubaker, who served as Interim CEO since September 29, 2014, continued in the post until Ms. Schneider joined us. Additionally, on September 29, 2014, the Board appointed Hassan Natha as CFO, and John Luttrell resigned as Interim CEO and CFO.

    On July 7, 2014, we received a notice from Lion asserting an event of default and an acceleration of the maturity of the loans and other outstanding obligations under the Lion Loan Agreement as a result of the suspension of Dov Charney as CEO by the Board. On July 14, 2014, Lion issued a notice rescinding the notice of acceleration. On July 16, 2014, Lion assigned its rights and obligations as a lender under the Lion Loan Agreement to an entity affiliated with Standard General. Standard General waived any default under the Standard General Loan Agreement that may have resulted or that might result from Mr. Charney not being the CEO.

    On September 8, 2014, we and Standard General entered into an amendment of the Standard General Loan Agreement to lower the applicable interest rate to 17%, extend the maturity to April 15, 2021, and make certain other technical amendments, including to remove a provision that specified that Mr. Charney not being the CEO would constitute an event of default.

    On March 25, 2015, we entered into the Sixth Amendment to the Capital One Credit Facility ("the Sixth Amendment") which (i) waived any defaults under the Capital One Credit Facility due to the failure to meet the obligation to maintain the maximum leverage ratio and minimum adjusted EBITDA required for the measurement periods ended December 31, 2014, as defined in the credit agreement, (ii) waived the obligation to maintain the minimum fixed charge coverage ratio, maximum leverage ratio and minimum adjusted EBITDA required for the twelve months ending March 31, 2015, (iii) included provisions to permit us to enter into the Standard General Credit Agreement, (iv) reset financial covenants relating to maintaining minimum fixed charge coverage ratios, maximum leverage ratios and minimum adjusted EBITDA and (v) permitted us to borrow $15,000 under the Standard General Credit Agreement.

    On March 25, 2015, one of our subsidiaries borrowed $15,000 under the Standard General Credit Agreement. The Standard General Credit Agreement is guaranteed by us, bears interest at 14% per annum, and will mature on October 15, 2020.

    In connection with the Standstill and Support Agreement among us, Standard General and Mr. Charney, five directors including Mr. Charney resigned from the Board effective as of August 2, 2014, and five new directors were appointed to the Board, three of whom were designated by Standard General and two of whom were appointed by the mutual agreement of Standard General and us. In addition, Lion exercised its rights to designate two members to our Board, whose appointments were effective as of September 15, 2014 and January 13, 2015, respectively. On March 6, 2015, a member appointed by Lion resigned from the Board, and on March 24, 2015, the Board elected a member designated by Lion to fill that vacancy.

    In 2012, German customs audited the import records of our German subsidiary for the years 2009 through 2011 and issued retroactive punitive duty assessments on certain containers of goods imported. The German customs imposed a substantially higher tariff rate than the original rate that we had paid on the imports, more than doubling the amount of the tariff that we would have to pay. The assessments of additional retaliatory duty originated from a trade dispute. Despite the ongoing appeals of the assessment, the German authorities demanded, and we paid, in connection with such assessment, $4,390 in the third quarter of 2014 and the final balance of $85 in the fourth quarter of 2014. We recorded the duty portion of $79 in cost of sales and the retaliatory duties, interest and penalties of $5,104 in general and administrative expenses in our consolidated statements of operations.

    Net sales for the year ended December 31, 2014 decreased $25,050, or 4.0%, from the year ended December 31, 2013 due to lower sales at our U.S. Retail, Canada and International segments, partly offset by an increase in the U.S. Wholesale segment.

    Gross profits as a percentage of sales were 50.8% and 50.6% for the year ended December 31, 2014 and 2013, respectively. Excluding the effects of the significant events described below, gross profits as a percentage of net sales increased slightly to 52.2% and 51.1% for the year ended December 31, 2014 and 2013, respectively. The increase was mainly due to a reduction in freight costs associated with the completion of our transition to the La Mirada distribution center in late 2013.

    Operating expenses for the year ended December 31, 2014 decreased $14,660, or 4.2%, from the year ended December 31, 2013. Excluding the effects of the significant events discussed below, operating expenses for the year ended December 31, 2014 decreased $27,616 from the year ended December 31, 2013. The decrease was primarily due to lower payroll from our cost reduction efforts and reduced expenditures on advertising and promotional activities.

    Loss from operations was $27,583 for the year ended December 31, 2014 as compared to $29,295 for the year ended December 31, 2013. Excluding the effects of the significant events discussed below, our operating results for the year ended December 31, 2014 would have been an income from operations of $6,838 as compared with a loss from operations of $13,482 for the year ended December 31, 2013. Lower operating expenses as discussed above were offset by lower sales volume and higher retail store impairments.

    Net loss for the year ended December 31, 2014 was $68,817 as compared to $106,298 for the year ended December 31, 2013. The improvement was mainly due to the $1,712 reduction in loss from operations due to the significant events discussed below, the change of $5,428 in fair value of warrants between periods, and the $32,101 loss on the extinguishment of debt in 2013. See Results of Operations for further details.

    Cash used in operating activities for the year ended December 31, 2014 was $5,212 compared to $12,723 for the year ended December 31, 2013 from the corresponding period in 2013. The decrease was mainly due to decreased inventory levels and improved operating income excluding certain significant costs discussed below. The decrease was partially offset by an increase in interest payments and payments related to the significant costs.

    Changes to Supply Chain Operations - In 2013, the transition to our new distribution center in La Mirada, California resulted in significant incremental costs (primarily labor). The issues surrounding the transition primarily related to improper design and integration and inadequate training and staffing. These issues caused processing inefficiencies that required us to employ additional staffing in order to meet customer demand. The transition was successfully completed during the fourth quarter of 2013. The center is now fully operational and labor costs have been reduced.

    Additional inventory reserves - In late 2014, new management undertook a strategic shift to change its inventory profile and actively reduce inventory levels to improve store merchandising, working capital and liquidity. As a result, we implemented an initiative to accelerate the sale of slow-moving inventory through our retail and online sales channels, as well as through certain off-price channels. As part of this process, management conducted a style-by-style review of inventory and identified certain slow-moving, second quality finished goods and raw materials inventories that required additional reserves as a result of the decision to accelerate sales of those items. Based on our analysis of the quantities on hand as well as the estimated recovery on these items, we significantly increased our excess and obsolescence reserve by $4,525 through a charge against cost of sales in our consolidated statements of operations.

    Customs settlements and contingencies - In 2012, German authorities audited the import records of our German subsidiary for the years 2009 through 2011 and issued retroactive punitive duty assessments on certain containers of goods imported. Despite ongoing appeals of the assessment, the German authorities demanded, and we paid, the outstanding balance of approximately $4,500 in the latter half of 2014. We recorded the duty portion of $79 in cost of sales and the retaliatory duties, interest and penalties of $5,104 in general and administrative expenses in our consolidated statements of operations. Additionally, during the fourth quarter of 2014, we wrote off approximately $3,300 in duty receivables to cost of sales in our consolidated statements of operations. These duty receivables related to changes in transfer costs for products sold to our European subsidiaries. We are also subject to, and have recorded charges related to, customs and similar audit settlements and contingencies in other jurisdictions.

    Internal Investigation - On June 18, 2014, the Board voted to replace Mr. Charney as Chairman of the Board, suspended him as our President and CEO and notified him of its intent to terminate his employment for cause. In connection with the Standstill and Support Agreement, the Board formed the Internal Investigation which ultimately concluded with his termination for cause on December 16, 2014. The suspension, internal investigation, and termination have resulted in substantial legal and consulting fees.

    Employment Settlements and Severance - In 2011, an industrial accident at our facility in Orange County, California resulted in a fatality to one of our employees, and in accordance with law, a mandatory criminal investigation was initiated. On August 19, 2014, a settlement of all claims related to the criminal investigation, pursuant to which the Company paid $1,000, was approved by the California Superior Court in Orange County. In addition, we had previously disclosed employment-related claims and experienced unusually high employee severance costs during 2014.

    (1) U.S. Wholesale

    U.S. Wholesale net sales for the year ended December 31, 2014, excluding online consumer net sales, increased by $8,113 or 5.1%, from the year ended December 31, 2013 mainly due to a significant new distributor that we added during the second quarter of 2014. We continue our focus on increasing our customer base by targeting direct sales, particularly sales to third-party screen printers. Online consumer net sales for the year ended December 31, 2014 decreased $395, or 1.0%, from the year ended December 31, 2013 mainly due to lower sales order volume. We continue our focus on targeted online advertising and promotional efforts.

    (2) U.S. Retail

    U.S. Retail net sales for the year ended December 31, 2014 decreased $13,569, or 6.6%, from the year ended December 31, 2013 mainly due to a decrease of approximately $14,000 in comparable store sales as a result of lower store foot traffic. Net sales decreased approximately $4,800 due to the closure of six stores in 2014, offset by an increase of approximately $1,100 from two new stores added since the beginning of January 2013.

    (3) Canada

    Canada net sales for the year ended December 31, 2014 decreased $8,590, or 14.3%, from the year ended December 31, 2013 mainly due to approximately $4,900 in lower sales, primarily in the retail and wholesale channels, and the unfavorable impact of foreign currency exchange rate changes of approximately $3,700.

    Retail net sales for the year ended December 31, 2014 decreased $7,076, or 15.7%, from the year ended December 31, 2013 due to $4,300 lower sales resulting from the closure of one retail store and approximately $1,700 from lower comparable store sales due to lower store foot traffic. Additionally, the impact of foreign currency exchange rate changes contributed to the sales decrease of approximately $2,800.

    Wholesale net sales for the year ended December 31, 2014 decreased $1,868, or 15.4%, from the year ended December 31, 2013. The decrease was largely due to lower sales orders resulting from a tightening focus on higher margin customers and lingering effects of order fulfillment delays associated with transition issues at the La Mirada distribution center. In addition, the impact of foreign currency exchange rate changes contributed to the sales decrease of approximately $700.

    Online consumer net sales for the year ended December 31, 2014 increased $354, or 12.3%, from the year ended December 31, 2013 mainly due to email advertising campaign, as well as improvements to the online store rolled out in the second half of 2013. This increase in sales was partially offset by the impact of foreign currency exchange rate changes of approximately $200.

    (4) International

    International net sales for the year ended December 31, 2014 decreased $10,609, or 6.3%, from the year ended December 31, 2013 due to approximately $10,500 lower sales in all three sales channels and the unfavorable impact of foreign currency exchange rate changes of approximately $100.

    Retail net sales for the year ended December 31, 2014 decreased $10,404, or 7.4%, from the year ended December 31, 2013. The decrease was due to lower comparable store sales of approximately $10,500 and lower sales of approximately $1,400 for the closure of five retail stores in 2014. The decrease was offset by approximately $200 higher sales due to seven new stores added since the beginning of January 2013 and the unfavorable impact of foreign currency exchange rate changes of approximately $400.

    Wholesale net sales for the year ended December 31, 2014 were flat as compared to the year ended December 31, 2013. The favorable impact of foreign currency exchange rate changes was approximately $100.

    Online consumer net sales for the year ended December 31, 2014 decreased $154, or 0.9%, from the year ended December 31, 2013 mainly due to lower sales order volume in Japan and Continental Europe, offset by higher sales order volume in Korea and the favorable impact of foreign currency exchange rate changes of approximately $200.

    (5) Gross profit

    Gross profit for the year ended December 31, 2014 decreased to $309,135 from $320,885 for the year ended December 31, 2013 due to lower retail sales volume at our U.S. Retail, Canada and International segments, offset by higher sales at our U.S. Wholesale segment. Excluding the effects of the significant events described above, gross profit as a percentage of net sales for the year ended December 31, 2014 slightly increased to 52.2% from 51.1%. The increase was mainly due to a decrease in freight costs associated with the completion of our transition to our La Mirada facility, offset by lower sales at our retail store operations.

    (6) Selling and distribution expenses

    Selling and distribution expenses for the year ended December 31, 2014 decreased $29,126, or 12.1%, from the year ended December 31, 2013. Excluding the effects of the changes to our supply chain operations discussed above, selling and distribution expenses decreased $17,279, or 7.5% from the year ended December 31, 2013 due primarily to lower selling related payroll costs of approximately $9,000, lower advertising costs of approximately $4,600 and lower travel and entertainment expenses of $1,400, all primarily as a result of our cost reduction efforts.

    (7) General and administrative expenses

    General and administrative expenses for the year ended December 31, 2014 increased $14,466, or 13.5%, from the year ended December 31, 2013. Excluding the effects of customs settlements and contingencies, the internal investigation, and employment settlements and severance discussed above, general and administrative expenses decreased $10,337, or 9.8% from the year ended December 31, 2013. The decrease was primarily due to $3,600 in lower share based compensation expense relating to the expiration and forfeiture of certain market based and performance based share awards and decreases in salaries and wages of approximately $3,800 and miscellaneous expenses such as travel, repair, and bank fees.

    (8) Loss from operations    

    Loss from operations was $27,583 for the year ended December 31, 2014 as compared to $29,295 for the year ended December 31, 2013. Excluding the effects of the significant events described above, our operating results for the year ended December 31, 2014 would have been an income from operations of $6,838 as compared with a loss from operations of $13,482 for the year ended December 31, 2013. Lower sales volume and higher retail store impairments were offset by decreases in our operating expenses as discussed above.

    (9) Income tax provision

    The provision for income tax for the year ended December 31, 2014 increased to $2,159 as compared to $1,771 for the year ended December 31, 2013. Although we incurred a loss from operations on a consolidated basis for the years ended December 31, 2014 and 2013, some of our foreign domiciled subsidiaries reported income from operations and are taxed on a stand-alone reporting basis. In 2014 and 2013, we recorded valuation allowances against a majority of our deferred tax assets, including 100% of the U.S. deferred tax assets and certain foreign deferred tax assets. We recognized no tax benefits on our loss before income taxes in 2014 and 2013.
    """,
    "Bed , Bath and Beyond": """
    Net sales decreased to $154.5 million for the three months ended November 2, 2013, compared to $188.1 million for the three months ended October 27, 2012. The decrease in net sales was primarily driven by a decrease in comparable premium retail store sales(1) of 16.8 percent, the impact of store closures and a decrease of 17.4 percent in our direct sales. 
    Gross profit was $48.2 million, or 31.2 percent of net sales, for the three months ended November 2, 2013, compared to $66.1 million, or 35.1 percent of net sales, for the three months ended October 27, 2012. The decrease in gross profit margin was primarily due to lower merchandise margins as a result of increased promotional activity and decreased leverage of buying and occupancy costs. 
    Selling, general and administrative expenses ("SG&A") were $70.8 million, or 45.9 percent of net sales, for the three months ended November 2, 2013, compared to $76.1 million, or 40.5 percent of net sales, for the three months ended October 27, 2012. The decrease of $5.3 million in SG&A was due to lower expenses across all categories with the largest declines from employee-related and marketing expenses.
    Net loss for the three months ended November 2, 2013 was $23.8 million, or $0.78 per share, and included other gain, net, of $8.0 million, or $0.26 per share, due to the change in the fair value of the derivative liability related to the Series A Preferred Stock, a severance charge of $2.3 million, or $0.07 per share, a non-cash impairment charge of $2.7 million, or $0.09 per share, and a non-cash income tax charge of $2.6 million, or $0.09 per share. This compares to a net loss for the three months ended October 27, 2012 of $20.5 million, or $0.67 per share, and included other loss, net, of $6.8 million, or $0.22 per share, due to the change in the fair value of the derivative liability related to the Series A Preferred Stock.
    We ended the third quarter of fiscal 2013 with $6.8 million in cash and cash equivalents compared to $31.3 million at the end of the third quarter of fiscal 2012. As of November 2, 2013, we had $15.0 million of borrowings outstanding under our revolving line of credit compared to no outstanding borrowings as of October 27, 2012. At the end of the third quarter of fiscal 2013, working capital was $6.6 million compared to $36.2 million at the end of the third quarter of fiscal 2012. Retail inventory per square foot, which includes inventory in our premium retail stores, factory stores, and in our distribution center, decreased 1.6 percent compared to the third quarter of fiscal 2012. Total inventory decreased 5.6 percent to $152.7 million at the end of the third quarter of fiscal 2013 from $161.7 million at the end of the third quarter of fiscal 2012. 
    Our turnaround has taken longer than expected and we have yet to experience consistent improvement in our sales. Despite the benefits from previous cost cutting initiatives and real estate optimization activities, we have continued to report significant operating losses. In addition to our merchandising and marketing initiatives discussed in more detail below, we announced a cost reduction program that is expected to generate $20.0 million to $25.0 million in pre-tax savings in fiscal year 2014. This program is designed to improve the financial and competitive position of the Company by streamlining the organization, reducing expenses, and positioning us for enhanced efficiency and profitability. Separately, we are evaluating approximately 70 potential lease actions for fiscal 2014 that could result in additional savings.
    For the remainder of fiscal 2013 and for fiscal 2014, we are focused on the following three critical initiatives that are foundational to achieving a successful turnaround:
    Our first initiative is focused on executing against our core assortment fundamentals. Our top five categories (knit tops, sweaters, woven tops, jeans and pants) drive the majority of our sales and profits and are key to our success. We are committed to organizational and process changes that will allow us to bring product to market in these five categories with a high degree of cross functional collaboration. We have made a number of changes to our product development process, which we believe will enable us to be more efficient as well as more consistent in our delivery of brand-edited, trend-right merchandise going forward. In addition, we are repositioning resources away from ancillary categories to heighten our focus on our top categories.
    The second initiative involves improving our brand perception. Under the leadership of our new Chief Marketing Officer, we are in the process of rolling out new marketing programs that are designed to create more excitement about our products, build our fashion credibility, improve our brand awareness, and grow our customer file. Through our new imagery and typography we will communicate the attributes of our products to our customers with a strong and consistent voice. We will build on the brand strategy we rolled out earlier this year to drive growth by significantly improving awareness and brand affinity through authentic brand differentiation, and by becoming a trustworthy style advisor and the leading destination for stylish apparel and accessories among our target audience of women. To achieve this, we will develop our brand aesthetic, voice, and express our role of trusted style advocate across all platforms and experiences in our business. We have been working rapidly to incorporate an elevated expression of our fashion and style advocacy, which began with the 2013 holiday season.
    Finally, our third initiative is focused on building customer relationships and loyalty. While we have a very loyal core customer, we are focused on targeting a broader customer base and leveraging all of our platforms, marketing strategies and marketing to attract and keep her. Ultimately, by growing our customer file and optimizing the existing customer base, we will improve brand engagement and traffic, increase market share, create loyalty, and inspire advocacy. We recently announced a new long-term agreement with Alliance Data Systems Corporation ("ADS") to expand our co-branded credit card and launch a private label credit card program. We will leverage ADS’s comprehensive suite of advanced analytics and multi-channel marketing capabilities to help strengthen our relationship with our customers.
    Other Developments
    In October 2013, the Board of Directors announced it would evaluate strategic alternatives to enhance value for stockholders. The Executive Committee of the Board of Directors is leading this process and intends to consider a broad range of alternatives, including, but not limited to, partnerships, joint ventures or a sale or merger of the Company. An independent financial advisor will assist the Board of Directors in the evaluation of possible strategic alternatives. There can be no assurance that the exploration of strategic alternatives will result in a transaction or that any transaction we enter into will prove to be beneficial to our stockholders.
    On July 26, 2013, we entered into a Credit Card Program Agreement (the "Program Agreement") with Comenity Bank, a bank subsidiary of ADS. Under the Program Agreement, ADS will issue co-branded credit cards and private label credit cards to approved new and existing customers. ADS will also purchase the existing co-branded credit card portfolio at a future date from Chase Bank USA, N.A. ("Chase"). During the nine months ended November 2, 2013, we received up-front incentive payments of $23.0 million, which was deferred and will be amortized over the term of the Program Agreement. We will receive an additional $2.0 million upon the launch of a new private label credit card program. We will also be entitled to future payments after ADS begins issuing credit cards under the Program Agreement for revenue sharing based on a percentage of credit card sales, certain new credit card accounts opened and activated, and profit sharing based on certain profitability measures of the program. The term of the Program Agreement is seven years from ADS's purchase of the co-branded credit card portfolio from Chase, which is expected to close in early fiscal 2014, with automatic extensions for successive one year terms.
    The $26.5 million decrease in retail segment net sales for the three months ended November 2, 2013 as compared to the three months ended October 27, 2012 is primarily the result of a 16.8 percent decrease in comparable premium retail store sales and the impact of 11 net store closures since the end of the third quarter of fiscal 2012. The decrease in comparable premium retail store sales was primarily driven by a 14.4 percent decrease in traffic and a 2.9 percent decrease in conversion.  
    The $7.1 million decrease in direct segment net sales for the three months ended November 2, 2013 as compared to the three months ended October 27, 2012 is primarily the result of lower order volume, partially offset by an increase in average transaction value and lower returns.
    Gross Profit
    Gross profit margin decreased by 3.9 percentage points during the three months ended November 2, 2013 as compared to the three months ended October 27, 2012. Gross profit margin was unfavorably impacted 2.6 percentage points from decreased leverage of buying and occupancy costs as well as lower merchandise margins due to increased promotional activity in the three months ended November 2, 2013 as compared to the three months ended October 27, 2012. 
    Selling, General and Administrative Expenses 
    SG&A expenses decreased $5.3 million during the three months ended November 2, 2013 as compared to the three months ended October 27, 2012, due to lower expenses across all categories with the largest declines from employee-related and marketing expenses.
    Loss on Asset Impairments
    During the three months ended November 2, 2013, we recorded impairment charges of $2.7 million related to certain long-lived assets, primarily premium store leasehold improvements. We did not have any impairment charges during the three months ended October 27, 2012.
    Retail segment operating income rate expressed as a percent of retail segment sales for the three months ended November 2, 2013 as compared to the three months ended October 27, 2012 decreased by 9.3 percentage points. The retail segment operating income rate was unfavorably impacted from decreased leverage of employee-related, occupancy and marketing expenses, primarily due to lower sales, as well as a $2.7 million impairment charge recorded during the quarter and 1.4 percentage points decline in margins.
    Direct segment operating income rate expressed as a percent of direct segment sales for the three months ended November 2, 2013 as compared to the three months ended October 27, 2012 decreased by 7.1 percentage points. The direct segment operating income rate was unfavorably impacted by 2.2 percentage points due to lower margins, as well as the unfavorable impact from decreased leverage of marketing and employee-related expenses primarily due to lower sales.
    Unallocated corporate and other expenses decreased $1.3 million for the three months ended November 2, 2013 as compared to the three months ended October 27, 2012, due to lower marketing and occupancy expenses, partially offset by higher employee-related expenses, primarily due to a severance charge of $2.3 million recorded in the current quarter.
    Other Loss (Gain), net 
    During the three months ended November 2, 2013, we recorded an $8.0 million gain from the fair value adjustment related to the derivative liability as compared to a $6.8 million loss from the fair value adjustment related to the derivative liability during the three months ended October 27, 2012.
    Interest Expense, net
    For the three months ended November 2, 2013, interest expense includes non-cash interest expense of $2.4 million as compared to $2.2 million for the same period in the prior year, which primarily includes accrued PIK interest expense and amortization of loan discounts and deferred debt issuance costs.
    Income Tax Provision
    The income tax provision primarily reflects the continuing impact of the valuation allowance against our net deferred tax assets, including an additional $2.6 million recorded in the current quarter on previously recorded net deferred tax assets, various state taxation requirements and certain discrete items.
    The $51.2 million decrease in retail segment net sales for the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012 is primarily the result of a 11.8 percent decrease in comparable premium retail store sales, the impact of 11 net store closures since the end of the third quarter of fiscal 2012, and a decrease of $5.3 million in net sales from factory stores, partially offset by an increase in credit card revenue recognized. The decrease in comparable premium retail store sales was primarily driven by a 9.8 percent decrease in traffic and a 4.8 percent decrease in comparable average transaction value, partially offset by a 3.0 percent increase in conversion.  
    The $10.5 million decrease in direct segment net sales for the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012 is primarily the result of a 19.3 percent decrease in order volume, partially offset by a 9.6 percent increase in average transaction value.
    Gross Profit
    Gross profit margin decreased by 1.3 percentage points during the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012 primarily due to a decrease of 0.8 percentage points in merchandise margins as a result of increased promotional activity and decreased leverage of buying and occupancy costs.
    Selling, General and Administrative Expenses 
    SG&A decreased $17.5 million during the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012, due to lower expenses across all categories with the largest declines from marketing and employee-related expenses.
    Loss on Asset Impairments
    During the nine months ended November 2, 2013, we recorded impairment charges of $2.7 million related to certain long-lived assets, primarily premium store leasehold improvements. We did not have any impairment charges during the nine months ended October 27, 2012.
    Retail segment operating income rate expressed as a percent of retail segment sales for the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012 decreased by 4.2 percentage points. The retail segment operating income rate was unfavorably impacted from decreased leverage of employee-related and occupancy expenses, primarily due to lower sales, as well as 1.5 percentage points decline in margins and the impact of a $2.7 million impairment charge recorded during fiscal 2013.   
    Direct segment operating income rate expressed as a percent of direct segment sales for the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012 increased by 2.0 percentage points. The direct segment operating income rate was favorably impacted by 2.4 percentage points due to higher margins as well as lower variable and fixed expenses, partially offset by the impact of higher marketing expenses as a percentage of sales.
    Unallocated corporate and other expenses decreased $5.4 million for the nine months ended November 2, 2013 as compared to the nine months ended October 27, 2012, primarily due to lower marketing, occupancy and employee-related expenses, partially offset by higher fixed expenses.
    Other Loss (Gain), net 
    During the nine months ended November 2, 2013, we recorded a $14.5 million gain from the fair value adjustment related to the derivative liability as compared to a $4.4 million loss from a fair value adjustment related to the derivative liability in addition to $1.1 million of issuance costs related to the Series A Preferred Stock recorded during the nine months ended October 27, 2012.
    Interest Expense, net
    The increase in interest expense, net, for the nine months ended November 2, 2013 as compared to the same period in the prior year is primarily the result of interest on borrowings under the Secured Term Loan that closed in the second quarter of fiscal 2012. For the nine months ended November 2, 2013, interest expense includes non-cash interest expense of $7.0 million as compared to $3.0 million for the same period in the prior year, which primarily includes accrued PIK interest expense and amortization of loan discounts and deferred debt issuance costs.
    Income Tax Provision
    The income tax provision primarily reflects the continuing impact of the valuation allowance against our net deferred tax assets, including an additional $2.6 million recorded in the current year on previously recorded net deferred tax assets, various state taxation requirements and certain discrete items.
    Seasonality 
    Our results of operations and cash flows have fluctuated, and will continue to fluctuate, on a quarterly basis, as well as on an annual basis, as a result of a number of factors, including, but not limited to, the following:


    •
    the composition, size and timing of various merchandise offerings;


    •
    the timing and number of premium retail store openings and closings;


    •
    the timing and number of promotions;


    •
    the timing and number of catalog mailings;


    •
    the ability to accurately estimate and accrue for merchandise returns and the costs of inventory disposition;


    •
    the timing of merchandise shipping and receiving, including any delays resulting from labor strikes or slowdowns, adverse weather conditions, health epidemics or national security measures; and


    •
    shifts in the timing of important holiday selling seasons relative to our fiscal quarters, including Valentine's Day, Easter, Mother's Day, Thanksgiving and Christmas, and the day of the week on which certain important holidays fall.
    Our results continue to depend materially on sales and profits from the November and December holiday shopping season. In anticipation of traditionally increased holiday sales activity, we incur certain significant incremental expenses, including the hiring of temporary employees to supplement the existing workforce.
    Liquidity and Capital Resources
    Overview
    Our turnaround has taken longer than expected and we have yet to experience consistent improvement in our sales. Despite the benefits from our previous merchandising and marketing initiatives, cost cutting initiatives and real estate optimization activities, we have continued to report significant net losses and negative year-to-date operating cash flows. As a result, we have implemented or plan to implement the following:



    In October 2013, the Board of Directors announced it would evaluate strategic alternatives to enhance value for stockholders. The Executive Committee of the Board of Directors is leading this process and intends to consider a broad range of alternatives, including, but not limited to, partnerships, joint ventures or a sale or merger of the Company. An independent financial advisor will assist the Board of Directors in the evaluation of possible strategic alternatives. There can be no assurance that the exploration of strategic alternatives will result in a transaction or that any transaction we enter into will prove to be beneficial to our stockholders.



    In October 2013, we announced that we are implementing further cost reduction initiatives that are expected to generate $20.0 million to $25.0 million in incremental pre-tax savings in fiscal year 2014. This program is designed to improve the financial and competitive position of the Company by streamlining the organization, reducing expenses, and positioning us for enhanced efficiency and profitability.



    We are evaluating approximately 70 store lease actions for fiscal 2014 that could result in additional savings. The expected outcome of these lease actions is for potential additional store closures, downsizing to smaller spaces, or amending leases at more favorable terms, all of which we expect would result in additional savings. We will continue to actively manage our store fleet to optimize our structure for the current environment.
    Our financial statements have been prepared on the basis that our business will continue as a going concern. We believe, based on our current projections, that we have sufficient sources of liquidity, including cash and cash equivalents and availability under our revolving line of credit, to fund our operations for at least the next twelve months. Our ability to fund our operations and to continue as a going concern depends upon meeting our projected future operating results, including the achievement of improvements from our merchandising and marketing initiatives, cost reduction program, store optimization program and other strategic initiatives, and the availability under our revolving line of credit, as well as the absence of any significant deterioration in consumer spending as a result of uncertain macroeconomic conditions. The ability to achieve our projected future operating results is based on a number of assumptions which involve significant judgment and estimates, which cannot be assured. If we are unable to achieve our projected operating results, we could violate one or more of our debt covenants, our liquidity could be adversely impacted and we may need to seek additional sources of liquidity. Our current level of debt could adversely affect our ability to raise additional capital to fund our operations and there is no assurance that debt or equity financing will be available in sufficient amounts or on acceptable terms. Therefore, a continuation of our recent historical operating results could result in our inability to continue as a going concern. Additional actions may include further reducing our expenditures, curtailing our operations, significantly restructuring our business, or restructuring our debt.
    We use our revolving line of credit to secure trade letters of credit and for borrowings, both of which reduce the amount of available borrowings. The actual amount that is available under our revolving line of credit fluctuates due to factors including, but not limited to, eligible inventory and credit card receivables, reserve amounts, outstanding letters of credit, and borrowing under our revolving line of credit. Consequently, it is possible that, should we need to access any additional funds from our revolving line of credit, it may not be available in full. 
    Operating Cash Flows
    Net cash used in operating activities decreased $11.9 million during the nine months ended November 2, 2013 as compared with the nine months ended October 27, 2012, primarily due to the receipt of $23.0 million in incentive payments related to our new Credit Card Program Agreement with ADS and the receipt of our annual revenue sharing payments under our current program as well as lower use of cash from operating assets and liabilities, partially offset by an increase in net loss, net of the non-cash activity.
    Investing Cash Flows
    Net cash used in investing activities principally consisted of cash outflows for capital expenditures which totaled $7.4 million and $14.2 million during the nine months ended November 2, 2013 and October 27, 2012, respectively. Capital expenditures during the nine months ended November 2, 2013 and October 27, 2012 primarily related to the relocation and remodeling of certain existing stores and the additional investment in our technology infrastructure.
    Financing Cash Flows
    Net cash provided by financing activities during the nine months ended November 2, 2013 primarily reflects net borrowings on our revolving line of credit. During the nine months ended October 27, 2012, net cash provided by financing activities primarily reflects the new Secured Term Loan with a portion of the proceeds used to pay down outstanding borrowings and related debt issuance costs.
    Secured Term Loan
    On July 9, 2012, we obtained a five-year, $65.0 million senior secured term loan (the "Secured Term Loan") provided by an affiliate of Golden Gate Capital. The Secured Term Loan bears interest at a rate of 5.5% to be paid in cash quarterly and 7.5% due and payable in kind ("PIK") upon maturity. The Secured Term Loan is collateralized by a second lien on our inventory and credit card receivables, and a first lien on our remaining assets. The Secured Term Loan is scheduled to mature upon the earlier of July 9, 2017 or the date that the obligations under the Amended and Restated Credit Agreement with Wells Fargo Bank dated May 16, 2011 (the "Credit Agreement") mature or are accelerated. Upon maturity (including as a result of an acceleration) of the Secured Term Loan, the principal balance and any unpaid interest, including $29.8 million of PIK interest, will become due
    Credit Agreement
    In May 2011, we entered into the Credit Agreement with a maturity date of May 16, 2016, which is secured primarily by our inventory, credit card receivables and certain other assets. The Credit Agreement provides a revolving line of credit of up to $70.0 million, with subfacilities for the issuance of up to $70.0 million in letters of credit and swingline advances of up to $10.0 million. The amount of credit that is available under the revolving line of credit is limited to a borrowing base that is determined according to, among other things, a percentage of the value of eligible inventory and credit card receivables, as reduced by certain reserve amounts required by Wells Fargo Bank. In conjunction with the closing of the Secured Term Loan, we amended the Credit Agreement and repaid the separate term loan previously provided by Wells Fargo Bank. The amendment did not materially change the terms of the Credit Agreement. As of November 2, 2013, the revolving line of credit was limited to a borrowing base of $70.0 million with $15.0 million in borrowings and $12.4 million in letters of credit issued, resulting in $42.6 million available for borrowing under our revolving line of credit.
    Pursuant to the Credit Agreement, borrowings issued under the revolving line of credit will generally accrue interest at a rate ranging from 1.00% to 2.50% (determined according to the average unused availability under the credit facility (the "Availability")) over a reference rate of, at our election, either LIBOR or a base rate (the "Reference Rate") with an interest rate of 2.20% as of November 2, 2013. Letters of credit issued under the revolving line of credit will accrue interest at a rate ranging from 1.50% to 2.50% (determined according to the Availability) with an interest rate of 2.00% as of November 2, 2013. Commitment fees accrue at a rate ranging from 0.375% to 0.50% (determined according to the Availability), which is assessed on the average unused portion of the credit facility maximum amount.
    The Secured Term Loan and Credit Agreement also contain various other covenants, such as capital expenditure limitations, restrictions on indebtedness, liens, investments, acquisitions, mergers, dispositions, dividends and other customary conditions. Our current store closure plans under our store optimization program and the related transfer or disposition of store assets is not limited by our Secured Term Loan or Credit Agreement.  Both the Secured Term Loan and Credit Agreement contain customary events of default. Upon an event of default that is not cured or waived within any applicable cure periods, in addition to other remedies that may be available to the lenders, the obligations may be accelerated, outstanding letters of credit may be required to be cash collateralized and remedies may be exercised against the collateral.
    Capital and Real Estate Plans
    Capital expenditures for fiscal 2013 are expected to be between $9.0 million to $10.0 million. We have limited new store openings to one factory store in fiscal 2013 and will continue to take advantage of real estate opportunities to improve the efficiency of our store base, including relocating stores to more favorable locations and reducing overall store size.
    In fiscal 2011, we announced a store optimization program where we expect to close up to 45 premium stores through fiscal 2013. By the end of fiscal 2013, we expect to close over 50 stores, including premium and factory stores and day spas. There is potential to close more under-performing stores beyond fiscal 2013. The optimization program is being achieved through a staged approach based primarily on natural lease expirations and early termination rights. We typically do not incur significant termination costs or disposal charges as a result of store closures. Early termination clauses generally relieve us of any future obligation under a lease if specified sales levels or certain occupancy targets are not achieved by a specified date. 
    Preparation of our consolidated financial statements in conformity with accounting principles generally accepted in the United States of America requires us to make estimates that affect the reported amounts of assets, liabilities, revenues and expenses, and the disclosure of contingent assets and liabilities. We base our estimates on historical experience and on other assumptions that we believe are reasonable. As a result, actual results could differ because of the use of estimates. The critical accounting policies used in the preparation of our consolidated financial statements include those that require us to make estimates about matters that are uncertain and could have a material impact to our consolidated financial statements. The description of critical accounting policies is included in our Annual Report on Form 10-K for the fiscal year ended February 2, 2013.
    """
}

def create_sentiment_flow_chart(sentence_results):
    """Create a combined sentiment and complexity chart for sentence-level analysis."""
    if not sentence_results:
        return None

    sentiment_scores = [s['final_sentiment_score'] for s in sentence_results]
    complexity_scores = [
        (len(s['valence_shifters']) / max(1, s['word_count'])) * 10
        for s in sentence_results
    ]
    sentence_indices = list(range(1, len(sentence_results) + 1))
    marker_colors = ['red' if score < 0 else 'green' if score > 0 else 'gray' for score in sentiment_scores]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            "Sentiment Flow Throughout Document",
            "Language Complexity by Sentence"
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=sentence_indices,
            y=sentiment_scores,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(color=marker_colors, size=8),
            hovertemplate='Sentence %{x}<br>Sentiment: %{y:.3f}<extra></extra>',
            name='Sentiment Score',
        ),
        row=1, col=1
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=sentence_indices,
            y=complexity_scores,
            marker_color='orange',
            hovertemplate='Sentence %{x}<br>Complexity: %{y:.2f}<extra></extra>',
            name='Complexity Score',
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        margin=dict(t=40, b=50, l=30, r=30),
        showlegend=False,
    )

    fig.update_xaxes(title_text="Sentence Number", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
    fig.update_yaxes(title_text="Complexity", row=2, col=1)

    return fig


# def create_risk_indicator_chart(risk_indicators_by_category):
#     """Create a pie chart showing risk indicators by category"""
#     if not any(risk_indicators_by_category.values()):
#         return None
   
#     categories = []
#     counts = []
#     colors = ['#FF6B6B', '#FF8E53', '#FF6B9D', '#4ECDC4', '#45B7D1']
   
#     for category, count in risk_indicators_by_category.items():
#         if count > 0:
#             categories.append(category.replace('_', ' ').title())
#             counts.append(count)
   
#     fig = go.Figure(data=[go.Pie(
#         labels=categories,
#         values=counts,
#         hole=0.4,
#         marker_colors=colors[:len(categories)]
#     )])
   
#     fig.update_layout(
#         title="Risk Indicators Distribution",
#         height=400
#     )
   
#     return fig

def create_complexity_gauge(complexity_score):
    """Create a gauge chart for complexity score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=complexity_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Complexity"},
        delta={'reference': 0.5},
        gauge={'axis': {'range': [None, 1]},
               'bar': {'color': "green"},
               'steps': [
                   {'range': [0, 0.3], 'color': "lightgray"},
                   {'range': [0.3, 0.7], 'color': "yellow"},
                   {'range': [0.7, 1], 'color': "red"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 0.9}}))
   
    fig.update_layout(height=300)
    return fig

def create_readability_chart(readability_metrics):
    """Create a bar chart for readability metrics"""
    metrics = {
        'Fog Index': readability_metrics['fog_index'],
        'Flesch-Kincaid': readability_metrics['flesch_kincaid'],
        'Avg Sentence Length': readability_metrics['avg_sentence_length'],
        'Complex Words %': readability_metrics['complex_words_ratio'] * 100
    }
   
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
        )
    ])
   
    fig.update_layout(
        title="Readability Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        height=400
    )
   
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Bankruptcy Risk Sentiment Analyzer</h1>', unsafe_allow_html=True)
   
    # Load analyzer
    analyzer = load_analyzer()
    if not analyzer:
        st.error("Failed to load the analyzer. Please check your model setup.")
        return
   
    # Sidebar
    st.sidebar.header("Analysis Options")
   
    # Company selection dropdown
    company_name = st.sidebar.selectbox(
        "Select Company",
        options=list(company_data.keys()),
        help="Choose a company to analyze its financial data"
    )
   
    # Analyze button
    if st.sidebar.button("Analyze Company", type="primary"):
        text_to_analyze = company_data.get(company_name, "")
        if text_to_analyze.strip():
            with st.spinner(f"Analyzing data for {company_name}..."):
                try:
                    result = analyzer.analyze_text(text_to_analyze)
                    st.session_state['analysis_result'] = result
                    st.session_state['company_name'] = company_name
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return
        else:
            st.warning("No data available for the selected company.")
   
    # Display results if available
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        company_name = st.session_state['company_name']
       
        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)
       
        with col1:
            sentiment_class = result['sentiment_classification']
            if sentiment_class == "Negative":
                st.metric("Sentiment Classification", "Negative")
            elif sentiment_class == "Positive":
                st.metric("Sentiment Classification", "Positive")
            else:
                st.metric("Sentiment Classification", "Neutral")
       
        with col2:
            bankruptcy_risk = result['bankruptcy_risk_score']
            if bankruptcy_risk == 1:
                st.metric("Bankruptcy Risk", "High Risk")
            elif bankruptcy_risk == 0:
                st.metric("Bankruptcy Risk", "Low Risk")
            else:
                risk_delta = "Medium Risk" if bankruptcy_risk > 0.3 else "Low Risk"
                st.metric("Bankruptcy Risk", f"{bankruptcy_risk:.3f}", delta=risk_delta)
       
        with col3:
            complexity = result.get('sentiment_complexity_score', 0)
            if complexity > 0.7:
                complexity_label = "High"
                complexity_class = "complexity-high"
            elif complexity > 0.4:
                complexity_label = "Medium"
                complexity_class = "complexity-medium"
            else:
                complexity_label = "Low"
                complexity_class = "complexity-low"
           
            st.metric("Complexity Score", f"{complexity:.3f}")
            st.markdown(f'<div class="{complexity_class}">Language Complexity: {complexity_label}</div>', unsafe_allow_html=True)
       
        with col4:
            st.metric("Risk Indicators", result['risk_indicators_count'])
            st.metric("Critical Risks", result['sentences_with_critical_risk'])
       
        # Readability and complexity row
        st.subheader("Document Readability & Complexity Analysis")
        col5, col6, col7 = st.columns(3)
       
        with col5:
            fog_index = result.get('fog_index', result['readability_metrics']['fog_index'])
            reading_level = "Graduate" if fog_index > 16 else "College" if fog_index > 12 else "High School" if fog_index > 8 else "Elementary"
            st.metric("Fog Index", f"{fog_index:.1f}", delta=reading_level)
            st.caption("Higher values = more complex text")
       
        with col6:
            fk_score = result.get('flesch_kincaid_score', result['readability_metrics']['flesch_kincaid'])
            st.metric("Flesch-Kincaid Grade", f"{fk_score:.1f}")
            st.caption("Grade level required to understand")
       
        with col7:
            avg_sentence_length = result['readability_metrics']['avg_sentence_length']
            length_assessment = "Very Long" if avg_sentence_length > 25 else "Long" if avg_sentence_length > 20 else "Medium" if avg_sentence_length > 15 else "Short"
            st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f} words", delta=length_assessment)
       
        # Charts section
        st.subheader("Visual Analysis")

        # First row of charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            flow_chart = create_sentiment_flow_chart(result['sentence_details'])
            if flow_chart:
                st.plotly_chart(flow_chart, use_container_width=True)

        with col_chart2:
            complexity_gauge = create_complexity_gauge(result.get('sentiment_complexity_score', 0))
            st.plotly_chart(complexity_gauge, use_container_width=True)

        # Second chart (readability) using full width since risk chart is removed
        readability_chart = create_readability_chart(result['readability_metrics'])
        st.plotly_chart(readability_chart, use_container_width=True)

        # Detailed analysis expandable sections
        st.subheader("Detailed Analysis")

        # Risk breakdown
        with st.expander("Risk Indicators Breakdown", expanded=False):
            if any(result['risk_indicators_by_category'].values()):
                st.write("**Risk Indicators by Category:**")
                risk_df = pd.DataFrame([
                    {"Category": category.replace('_', ' ').title(), "Count": count}
                    for category, count in result['risk_indicators_by_category'].items()
                    if count > 0
                ])
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.info("No risk indicators detected in this document.")

        # Document statistics
        # with st.expander("Document Statistics", expanded=False):
        #     col_stat1, col_stat2, col_stat3 = st.columns(3)

        #     with col_stat1:
        #         st.metric("Total Sentences", result['total_sentences_analyzed'])
        #         st.metric("Risk Flagged Sentences", result['sentences_with_risk_flags'])

        #     with col_stat2:
        #         st.metric("Economic Headwinds", result['sentences_with_economic_headwinds'])
        #         st.metric("Valence Shifters", result['valence_shifter_frequency'])

        #     with col_stat3:
        #         st.metric("Sentiment Std Dev", f"{result['sentiment_std']:.3f}")
        #         st.metric("Sentiment Range", f"{result['sentiment_range']:.3f}")

       
        st.title(" Company News Dashboard")

        # # Dropdown to select company
        # companies = ['Ascena Retail Group', 'American Apparel', "Bed, Bath and Beyond"]
        # company_name = st.selectbox("Select a Company to View News", companies)

        # Company news section (main area)
        st.subheader(f"Latest News: {company_name}")

        try:
            news_articles = get_company_news(company_name)
            for i, article in enumerate(news_articles):
                with st.container():
                    col_news1, col_news2 = st.columns([3, 1])
                    with col_news1:
                        st.markdown(f"**{article['title']}**")
                        st.write(article['description'])
                        st.caption(f"Source: {article['source']} | Published: {article['publishedAt'][:10]}")
                    with col_news2:
                        if st.button(f"Read Article {i+1}", key=f"news_{i}"):
                            st.markdown(f"[Open Article]({article['url']})", unsafe_allow_html=True)
                st.divider()
        except Exception as e:
            st.info("News feature requires API setup. Currently showing sample data for demonstration.")
   
    else:
        # Welcome message and instructions
        st.markdown("## Item-7 (MD&A) Sentiment Analysis")

       
if __name__ == "__main__":
    main()