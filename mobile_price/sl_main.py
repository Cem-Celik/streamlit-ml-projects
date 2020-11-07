#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sl_mobile_price
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

@st.cache
def main():
    st.title('Projects')
    set_image()
    if st.button("Start"):
        sl_mobile_price.main()       

def call_price():
    sl_mobile_price.main()



if __name__ == "__main__":
    main()