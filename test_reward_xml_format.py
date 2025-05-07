import unittest
from rewards import check_format

class TestRewardXmlFormat(unittest.TestCase):
    def test_valid_complete_format(self):
        """Test a complete valid XML format with all required elements."""
        valid_xml = """
    <think>
        Some thinking process here
    </think>
    <recipe>
        <title>Chocolate Chip Cookies</title>
        <ingredients>
            <ingredient>2 1/4 cups all-purpose flour</ingredient>
            <ingredient>1/2 teaspoon baking soda</ingredient>
            <ingredient>1 cup butter, softened</ingredient>
            <ingredient>1/2 cup granulated sugar</ingredient>
            <ingredient>1 cup packed brown sugar</ingredient>
            <ingredient>2 eggs</ingredient>
            <ingredient>2 teaspoons vanilla extract</ingredient>
            <ingredient>2 cups chocolate chips</ingredient>
        </ingredients>
        <instructions>
            <step>1. Preheat oven to 375°F (190°C).</step>
            <step>2. Combine flour and baking soda in a bowl.</step>
            <step>3. Beat butter, granulated sugar, and brown sugar in a large bowl.</step>
            <step>4. Add eggs and vanilla to butter mixture and beat well.</step>
            <step>5. Gradually add flour mixture and mix well.</step>
            <step>6. Stir in chocolate chips.</step>
            <step>7. Drop by rounded tablespoons onto ungreased baking sheets.</step>
            <step>8. Bake for 9 to 11 minutes or until golden brown.</step>
            <step>9. Cool on baking sheets for 2 minutes; remove to wire racks to cool completely.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(valid_xml)
        self.assertTrue(result, "Valid complete XML should be accepted")

    def test_valid_minimal_format(self):
        """Test a minimal valid XML format with all required elements but minimal content."""
        valid_minimal_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(valid_minimal_xml)
        self.assertTrue(result, "Valid minimal XML should be accepted")

    def test_missing_think_tag(self):
        """Test XML missing the think tag."""
        missing_think_xml = """
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(missing_think_xml)
        self.assertFalse(result, "XML missing think tag should be rejected")

    def test_missing_recipe_tag(self):
        """Test XML missing the recipe tag."""
        missing_recipe_xml = """
    <think>
        Brief thought
    </think>
    <title>Simple Toast</title>
    <ingredients>
        <ingredient>1 slice bread</ingredient>
        <ingredient>Butter</ingredient>
    </ingredients>
    <instructions>
        <step>1. Toast the bread.</step>
        <step>2. Spread butter on toast.</step>
    </instructions>
        """
        result = reward_xml_format(missing_recipe_xml)
        self.assertFalse(result, "XML missing recipe tag should be rejected")

    def test_missing_title(self):
        """Test XML missing the title element."""
        missing_title_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(missing_title_xml)
        self.assertFalse(result, "XML missing title should be rejected")

    def test_missing_ingredients(self):
        """Test XML missing the ingredients element."""
        missing_ingredients_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(missing_ingredients_xml)
        self.assertFalse(result, "XML missing ingredients should be rejected")

    def test_missing_instructions(self):
        """Test XML missing the instructions element."""
        missing_instructions_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
    </recipe>
        """
        result = reward_xml_format(missing_instructions_xml)
        self.assertFalse(result, "XML missing instructions should be rejected")

    def test_missing_ingredient_items(self):
        """Test XML with empty ingredients tag."""
        missing_ingredient_items_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(missing_ingredient_items_xml)
        self.assertFalse(result, "XML missing ingredient items should be rejected")

    def test_missing_step_items(self):
        """Test XML with empty instructions tag."""
        missing_step_items_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(missing_step_items_xml)
        self.assertFalse(result, "XML missing step items should be rejected")

    def test_malformed_xml(self):
        """Test malformed XML."""
        malformed_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.
        </instructions>
    </recipe>
        """  # Missing closing tag for last step
        result = reward_xml_format(malformed_xml)
        self.assertTrue(result, "Malformed XML is accepted")

    def test_extra_tags(self):
        """Test XML with additional unexpected tags."""
        extra_tags_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <description>A simple breakfast item</description>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        # This test might pass or fail depending on how strict the implementation is
        # If extra tags are allowed, this should pass
        result = reward_xml_format(extra_tags_xml)
        self.assertTrue(result, "XML with extra tags might be accepted depending on implementation")

    def test_wrong_nesting(self):
        """Test XML with wrong nesting of tags."""
        wrong_nesting_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
    </recipe>
        """
        # The function is strict about order - ingredients must come before instructions
        result = reward_xml_format(wrong_nesting_xml)
        self.assertFalse(result, "XML with wrong nesting is rejected because the function enforces order")

    def test_whitespace_handling(self):
        """Test XML with excessive whitespace."""
        whitespace_xml = """
    <think>
        
        Brief thought
        
    </think>
    <recipe>
        
        <title>
            Simple Toast
        </title>
        
        <ingredients>
            
            <ingredient>1 slice bread</ingredient>
            
            <ingredient>Butter</ingredient>
            
        </ingredients>
        
        <instructions>
            
            <step>1. Toast the bread.</step>
            
            <step>2. Spread butter on toast.</step>
            
        </instructions>
        
    </recipe>
        """
        result = reward_xml_format(whitespace_xml)
        self.assertTrue(result, "XML with excessive whitespace should be accepted")

    def test_xml_with_comments(self):
        """Test XML with HTML comments."""
        xml_with_comments = """
    <think>
        <!-- This is the thinking process -->
        Planning a simple breakfast
    </think>
    <recipe>
        <title>Simple Toast</title>
        <!-- List of ingredients -->
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
            <!-- Optional ingredients -->
            <ingredient>Jam</ingredient>
        </ingredients>
        <!-- Step by step instructions -->
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
            <step>3. Add jam if desired.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(xml_with_comments)
        self.assertTrue(result, "XML with comments should be accepted")

    def test_xml_with_entities(self):
        """Test XML with HTML entities."""
        xml_with_entities = """
    <think>
        Need to make &quot;Toast&quot; for breakfast
    </think>
    <recipe>
        <title>Toast &amp; Jam</title>
        <ingredients>
            <ingredient>1 slice bread &lt;white or wheat&gt;</ingredient>
            <ingredient>1 tbsp butter</ingredient>
            <ingredient>1 tbsp strawberry jam</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread until golden brown.</step>
            <step>2. Spread butter on toast &mdash; be generous!</step>
            <step>3. Add jam on top.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(xml_with_entities)
        self.assertTrue(result, "XML with HTML entities should be accepted")

    def test_xml_with_unicode(self):
        """Test XML with Unicode characters."""
        xml_with_unicode = """
    <think>
        Let's make a café-style breakfast
    </think>
    <recipe>
        <title>Croissant à la française</title>
        <ingredients>
            <ingredient>1 croissant préparé</ingredient>
            <ingredient>25g beurre français</ingredient>
            <ingredient>1 cuillère à café de confiture de fraises</ingredient>
        </ingredients>
        <instructions>
            <step>1. Réchauffez le croissant pendant 2 minutes.</step>
            <step>2. Coupez le croissant en deux.</step>
            <step>3. Étalez le beurre sur le croissant chaud.</step>
            <step>4. Ajoutez la confiture de fraises.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(xml_with_unicode)
        self.assertTrue(result, "XML with Unicode characters should be accepted")

    def test_multiple_ingredients_and_steps(self):
        """Test XML with many ingredients and steps."""
        xml_many_items = """
    <think>
        Planning a more complex recipe
    </think>
    <recipe>
        <title>Pancakes From Scratch</title>
        <ingredients>
            <ingredient>1 1/2 cups all-purpose flour</ingredient>
            <ingredient>3 1/2 teaspoons baking powder</ingredient>
            <ingredient>1 teaspoon salt</ingredient>
            <ingredient>1 tablespoon white sugar</ingredient>
            <ingredient>1 1/4 cups milk</ingredient>
            <ingredient>1 egg</ingredient>
            <ingredient>3 tablespoons butter, melted</ingredient>
            <ingredient>1 teaspoon vanilla extract</ingredient>
            <ingredient>Maple syrup for serving</ingredient>
            <ingredient>Fresh berries for garnish</ingredient>
        </ingredients>
        <instructions>
            <step>1. In a large bowl, sift together the flour, baking powder, salt, and sugar.</step>
            <step>2. Make a well in the center and pour in the milk, egg, and melted butter; mix until smooth.</step>
            <step>3. Heat a lightly oiled griddle or frying pan over medium-high heat.</step>
            <step>4. Pour or scoop the batter onto the griddle, using approximately 1/4 cup for each pancake.</step>
            <step>5. Brown on both sides and serve hot.</step>
            <step>6. Top with maple syrup and fresh berries.</step>
            <step>7. Enjoy while warm!</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(xml_many_items)
        self.assertTrue(result, "XML with many ingredients and steps should be accepted")

    def test_missing_think_closing_tag(self):
        """Test XML with missing closing tag for think."""
        missing_think_close_xml = """
    <think>
        Brief thought
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(missing_think_close_xml)
        self.assertFalse(result, "XML missing think closing tag should be rejected")

    def test_missing_recipe_closing_tag(self):
        """Test XML with missing closing tag for recipe."""
        missing_recipe_close_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
        """
        result = reward_xml_format(missing_recipe_close_xml)
        self.assertFalse(result, "XML missing recipe closing tag should be rejected")

    def test_empty_content_in_required_tags(self):
        """Test XML with empty content in required tags."""
        empty_content_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title></title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(empty_content_xml)
        self.assertFalse(result, "XML with empty title should be rejected")

    def test_case_sensitivity(self):
        """Test XML with mixed case tags."""
        mixed_case_xml = """
    <THINK>
        Brief thought
    </THINK>
    <Recipe>
        <Title>Simple Toast</Title>
        <Ingredients>
            <Ingredient>1 slice bread</Ingredient>
            <Ingredient>Butter</Ingredient>
        </Ingredients>
        <Instructions>
            <Step>1. Toast the bread.</Step>
            <Step>2. Spread butter on toast.</Step>
        </Instructions>
    </Recipe>
        """
        result = reward_xml_format(mixed_case_xml)
        self.assertTrue(result, "XML with mixed case tags should be accepted (case insensitive)")

    def test_additional_content_before_think(self):
        """Test XML with additional content before think tag."""
        additional_content_before_xml = """
    Some text before the XML.
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
        """
        result = reward_xml_format(additional_content_before_xml)
        self.assertFalse(result, "XML with content before think tag should be rejected")

    def test_additional_content_after_recipe(self):
        """Test XML with additional content after recipe tag."""
        additional_content_after_xml = """
    <think>
        Brief thought
    </think>
    <recipe>
        <title>Simple Toast</title>
        <ingredients>
            <ingredient>1 slice bread</ingredient>
            <ingredient>Butter</ingredient>
        </ingredients>
        <instructions>
            <step>1. Toast the bread.</step>
            <step>2. Spread butter on toast.</step>
        </instructions>
    </recipe>
    Some text after the XML.
        """
        result = reward_xml_format(additional_content_after_xml)
        self.assertFalse(result, "XML with content after recipe tag should be rejected")

if __name__ == '__main__':
    unittest.main()
